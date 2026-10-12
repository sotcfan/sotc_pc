#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "loader/self.h"

namespace loader {

// A symbol the module takes from a Sony library.
struct ImportRef {
    std::string nid;
    std::string library; // as named by the module, e.g. "libkernel"
    bool is_object;      // a variable rather than a function
};

// A PS4 executable mapped into this process at a fixed address.
class Module {
public:
    struct TlsInfo {
        u64 image = 0; // initial contents
        u64 init_size = 0;
        u64 size = 0;
        u64 align = 0;
    };

    // Maps every segment at `base`. Throws std::runtime_error if the range is taken.
    Module(std::string name, const ElfFile& elf, u64 base);

    const std::string& Name() const {
        return name_;
    }
    u64 Base() const {
        return base_;
    }
    u64 Size() const {
        return size_;
    }
    bool Contains(u64 address) const {
        return address >= base_ && address < base_ + size_;
    }
    u64 EntryAddress() const {
        return entry_;
    }
    const ProcParam* GetProcParam() const {
        return reinterpret_cast<const ProcParam*>(proc_param_);
    }
    const TlsInfo& Tls() const {
        return tls_;
    }

    // Applies every relocation. `resolve` gives the address an imported symbol must point to;
    // it is called once per symbol.
    void Relocate(const std::function<u64(const ImportRef&)>& resolve);

    // Gives each segment its final permissions. Call after Relocate().
    void Protect();

    // Code segments as (address, size); writable until Protect().
    std::vector<std::pair<u64, u64>> ExecutableRanges() const;

private:
    struct Segment {
        u64 address;
        u64 size;
        u32 flags; // PF_R / PF_W / PF_X
    };

    void ParseDynamic(const ElfFile& elf);
    const char* String(u64 offset) const;
    ImportRef MakeImport(const Elf64_Sym& symbol) const;

    std::string name_;
    u64 base_ = 0;
    u64 size_ = 0;
    u64 entry_ = 0;
    u64 proc_param_ = 0;
    TlsInfo tls_;
    std::vector<Segment> segments_;

    // Copy of PT_SCE_DYNLIBDATA and the tables inside it.
    std::vector<u8> dynlib_data_;
    u64 strtab_offset_ = 0, strtab_size_ = 0;
    u64 symtab_offset_ = 0, symtab_size_ = 0;
    u64 rela_offset_ = 0, rela_size_ = 0;
    u64 jmprel_offset_ = 0, jmprel_size_ = 0;
    std::map<u64, std::string> libraries_; // library id -> name
};

} // namespace loader
