#include "common/host_memory.h"
#include "common/platform.h"
#include "loader/module.h"


#include <cerrno>
#include <cstring>
#include <format>
#include <stdexcept>
#include <unordered_map>

#include "common/log.h"
#include "loader/nid.h"

namespace loader {

namespace {

// PS4 pages are 16 KiB; segments are aligned to that.
constexpr u64 PageSize = 0x4000;

constexpr u64 AlignDown(u64 value, u64 align) {
    return value & ~(align - 1);
}
constexpr u64 AlignUp(u64 value, u64 align) {
    return AlignDown(value + align - 1, align);
}

bool IsMapped(const Elf64_Phdr& ph) {
    return (ph.p_type == PT_LOAD || ph.p_type == PT_SCE_RELRO) && ph.p_memsz != 0;
}

int ToProt(u32 flags) {
    int prot = host_memory::ProtNone;
    if (flags & PF_R) {
        prot |= host_memory::ProtRead;
    }
    if (flags & PF_W) {
        prot |= host_memory::ProtWrite;
    }
    if (flags & PF_X) {
        prot |= host_memory::ProtExec;
    }
    return prot;
}

} // namespace

Module::Module(std::string name, const ElfFile& elf, u64 base) : name_{std::move(name)}, base_{base} {
    for (const auto& ph : elf.phdrs) {
        if (IsMapped(ph)) {
            size_ = std::max(size_, AlignUp(ph.p_vaddr + ph.p_memsz, PageSize));
        }
    }
    if (size_ == 0) {
        throw std::runtime_error(std::format("{}: no segment to load", name_));
    }

    // Everything starts writable so relocations can be applied; Protect() tightens it.
    void* mapped = host_memory::AllocateAt(base_, size_,
                                           host_memory::ProtRead | host_memory::ProtWrite);
    if (!mapped) {
        throw std::runtime_error(std::format("{}: could not map {:#x}-{:#x}: {}", name_,
                                             base_, base_ + size_, platform::LastErrorText()));
    }

    for (const auto& ph : elf.phdrs) {
        if (IsMapped(ph)) {
            if (ph.p_offset + ph.p_filesz > elf.data.size()) {
                throw std::runtime_error(std::format("{}: segment outside the file", name_));
            }
            std::memcpy(reinterpret_cast<void*>(base_ + ph.p_vaddr), elf.data.data() + ph.p_offset,
                        ph.p_filesz);
            segments_.push_back({base_ + ph.p_vaddr, ph.p_memsz, ph.p_flags});
        } else if (ph.p_type == PT_TLS) {
            tls_ = {base_ + ph.p_vaddr, ph.p_filesz, ph.p_memsz, ph.p_align};
        } else if (ph.p_type == PT_SCE_PROCPARAM) {
            proc_param_ = base_ + ph.p_vaddr;
        }
    }
    entry_ = base_ + elf.header.e_entry;

    ParseDynamic(elf);
    LOG_INFO("Loader", "{} loaded at {:#x}-{:#x}, entry {:#x}", name_, base_,
             base_ + size_, entry_);
}

void Module::ParseDynamic(const ElfFile& elf) {
    const Elf64_Phdr* dynamic = elf.FindSegment(PT_DYNAMIC);
    const Elf64_Phdr* dynlib = elf.FindSegment(PT_SCE_DYNLIBDATA);
    if (!dynamic || !dynlib) {
        throw std::runtime_error(std::format("{}: no dynamic table", name_));
    }
    dynlib_data_.assign(elf.data.begin() + dynlib->p_offset,
                        elf.data.begin() + dynlib->p_offset + dynlib->p_filesz);

    std::vector<u64> library_values;
    for (u64 offset = 0; offset + sizeof(Elf64_Dyn) <= dynamic->p_filesz;
         offset += sizeof(Elf64_Dyn)) {
        Elf64_Dyn dyn;
        std::memcpy(&dyn, elf.data.data() + dynamic->p_offset + offset, sizeof(dyn));
        if (dyn.d_tag == DT_NULL) {
            break;
        }
        const u64 value = dyn.d_un.d_val;
        switch (dyn.d_tag) {
        case DT_SCE_STRTAB: strtab_offset_ = value; break;
        case DT_SCE_STRSZ: strtab_size_ = value; break;
        case DT_SCE_SYMTAB: symtab_offset_ = value; break;
        case DT_SCE_SYMTABSZ: symtab_size_ = value; break;
        case DT_SCE_RELA: rela_offset_ = value; break;
        case DT_SCE_RELASZ: rela_size_ = value; break;
        case DT_SCE_JMPREL: jmprel_offset_ = value; break;
        case DT_SCE_PLTRELSZ: jmprel_size_ = value; break;
        case DT_SCE_IMPORT_LIB:
        case DT_SCE_EXPORT_LIB: library_values.push_back(value); break;
        default: break;
        }
    }

    const auto in_blob = [this](u64 offset, u64 size) {
        return offset <= dynlib_data_.size() && size <= dynlib_data_.size() - offset;
    };
    if (!in_blob(strtab_offset_, strtab_size_) || !in_blob(symtab_offset_, symtab_size_) ||
        !in_blob(rela_offset_, rela_size_) || !in_blob(jmprel_offset_, jmprel_size_)) {
        throw std::runtime_error(std::format("{}: corrupted dynamic table", name_));
    }

    // Library entries: name offset in the low 32 bits, id in the top 16.
    for (u64 value : library_values) {
        libraries_[value >> 48] = String(value & 0xFFFFFFFF);
    }
}

const char* Module::String(u64 offset) const {
    if (offset >= strtab_size_) {
        return "";
    }
    return reinterpret_cast<const char*>(dynlib_data_.data() + strtab_offset_ + offset);
}

ImportRef Module::MakeImport(const Elf64_Sym& symbol) const {
    // Imported names look like "NID#library-id#module-id".
    const std::string_view full = String(symbol.st_name);
    ImportRef ref;
    ref.is_object = ELF64_ST_TYPE(symbol.st_info) == STT_OBJECT;
    const size_t first = full.find('#');
    ref.nid = std::string(full.substr(0, first));
    if (first != std::string_view::npos) {
        const size_t second = full.find('#', first + 1);
        u64 id;
        if (DecodeId(full.substr(first + 1, second - first - 1), id)) {
            if (const auto it = libraries_.find(id); it != libraries_.end()) {
                ref.library = it->second;
            }
        }
    }
    return ref;
}

void Module::Relocate(const std::function<u64(const ImportRef&)>& resolve) {
    const auto* symbols = reinterpret_cast<const Elf64_Sym*>(dynlib_data_.data() + symtab_offset_);
    const u64 symbol_count = symtab_size_ / sizeof(Elf64_Sym);
    std::unordered_map<u64, u64> resolved; // symbol index -> address

    const auto symbol_address = [&](u64 index) -> u64 {
        if (index >= symbol_count) {
            throw std::runtime_error(std::format("{}: relocation with an invalid symbol", name_));
        }
        const Elf64_Sym& symbol = symbols[index];
        if (ELF64_ST_BIND(symbol.st_info) == STB_LOCAL) {
            return base_ + symbol.st_value;
        }
        auto [it, inserted] = resolved.try_emplace(index, 0);
        if (inserted) {
            it->second = resolve(MakeImport(symbol));
        }
        return it->second;
    };

    u64 counts[3] = {};
    const auto apply = [&](u64 table_offset, u64 table_size) {
        for (u64 offset = 0; offset + sizeof(Elf64_Rela) <= table_size;
             offset += sizeof(Elf64_Rela)) {
            Elf64_Rela rela;
            std::memcpy(&rela, dynlib_data_.data() + table_offset + offset, sizeof(rela));
            if (!Contains(base_ + rela.r_offset) || rela.r_offset + 8 > size_) {
                throw std::runtime_error(std::format("{}: relocation outside the image", name_));
            }
            u64 value;
            switch (ELF64_R_TYPE(rela.r_info)) {
            case R_X86_64_RELATIVE:
                value = base_ + rela.r_addend;
                counts[0]++;
                break;
            case R_X86_64_64:
                value = symbol_address(ELF64_R_SYM(rela.r_info)) + rela.r_addend;
                counts[1]++;
                break;
            case R_X86_64_GLOB_DAT:
            case R_X86_64_JUMP_SLOT:
                value = symbol_address(ELF64_R_SYM(rela.r_info));
                counts[2]++;
                break;
            default:
                throw std::runtime_error(std::format("{}: unsupported relocation type {}",
                                                     name_, ELF64_R_TYPE(rela.r_info)));
            }
            std::memcpy(reinterpret_cast<void*>(base_ + rela.r_offset), &value, sizeof(value));
        }
    };
    apply(rela_offset_, rela_size_);
    apply(jmprel_offset_, jmprel_size_);
    LOG_INFO("Loader", "{}: {} relative relocations, {} with a symbol, {} for imports", name_,
             counts[0], counts[1], counts[2]);
}

std::vector<std::pair<u64, u64>> Module::ExecutableRanges() const {
    std::vector<std::pair<u64, u64>> ranges;
    for (const auto& segment : segments_) {
        if (segment.flags & PF_X) {
            ranges.emplace_back(segment.address, segment.size);
        }
    }
    return ranges;
}

void Module::Protect() {
    for (const auto& segment : segments_) {
        const u64 start = AlignDown(segment.address, PageSize);
        const u64 end = AlignUp(segment.address + segment.size, PageSize);
        if (!host_memory::Protect(reinterpret_cast<void*>(start), end - start,
                                  ToProt(segment.flags))) {
            throw std::runtime_error(std::format("{}: protection failed at {:#x}: {}", name_,
                                                 start, platform::LastErrorText()));
        }
    }
}

} // namespace loader
