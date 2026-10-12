// Function directives for both object formats: ELF (Linux) and COFF (Windows).
#ifdef _WIN32
#define FUNCTION_BEGIN(name) .globl name; .def name; .scl 2; .type 32; .endef; name:
#define FUNCTION_END(name)
#else
#define FUNCTION_BEGIN(name) .globl name; .type name, @function; name:
#define FUNCTION_END(name) .size name, . - name
#endif

    .intel_syntax noprefix
    .text

// void fiber_switch(u64* save_rsp, u64 resume_rsp, u32* release_state, u32 release_value)
// Saves the callee-saved registers, MXCSR and the x87 control word of the current context on its
// stack, stores the stack pointer in *save_rsp, and resumes the context saved at resume_rsp.
// Only once the context is saved, it stores release_value in *release_state (if not null): that
// is what lets other threads resume the context being left, so it must not happen earlier.
// The saved frame, from the lowest address: mxcsr (4 bytes), x87 control word (2), padding (2),
// r15, r14, r13, r12, rbx, rbp, return address.
    FUNCTION_BEGIN(fiber_switch)
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    stmxcsr [rsp]
    fnstcw [rsp + 4]
    mov [rdi], rsp
    test rdx, rdx
    jz 1f
    mov [rdx], ecx
1:
    mov rsp, rsi
    ldmxcsr [rsp]
    fldcw [rsp + 4]
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    FUNCTION_END(fiber_switch)

// First code a new fiber runs: the frame built by PrepareStack() puts the fiber in r12.
    FUNCTION_BEGIN(fiber_start)
    mov rdi, r12
    and rsp, -16
    call FiberMain
    ud2
    FUNCTION_END(fiber_start)

#ifndef _WIN32
    .section .note.GNU-stack, "", @progbits
#endif
