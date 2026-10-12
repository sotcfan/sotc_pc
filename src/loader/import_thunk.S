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

// Every import that is not bound directly jumps here from its stub, with r11 = ImportInfo*.
// The argument registers are saved, OnImportCall() records the call and returns the address to
// continue at, and the registers are restored before jumping there. The caller's stack is left
// untouched, so arguments passed on the stack still reach the real function.
    FUNCTION_BEGIN(import_thunk)
    push rbp
    mov rbp, rsp
    and rsp, -16
    sub rsp, 192
    mov [rsp + 0], rdi
    mov [rsp + 8], rsi
    mov [rsp + 16], rdx
    mov [rsp + 24], rcx
    mov [rsp + 32], r8
    mov [rsp + 40], r9
    mov [rsp + 48], rax          // al = vector registers used by a variadic call
    mov [rsp + 56], r10
    movdqu [rsp + 64], xmm0
    movdqu [rsp + 80], xmm1
    movdqu [rsp + 96], xmm2
    movdqu [rsp + 112], xmm3
    movdqu [rsp + 128], xmm4
    movdqu [rsp + 144], xmm5
    movdqu [rsp + 160], xmm6
    movdqu [rsp + 176], xmm7

    mov rdi, r11
    mov rsi, [rbp + 8]           // return address inside the caller
    call OnImportCall
    mov r11, rax

    mov rdi, [rsp + 0]
    mov rsi, [rsp + 8]
    mov rdx, [rsp + 16]
    mov rcx, [rsp + 24]
    mov r8, [rsp + 32]
    mov r9, [rsp + 40]
    mov rax, [rsp + 48]
    mov r10, [rsp + 56]
    movdqu xmm0, [rsp + 64]
    movdqu xmm1, [rsp + 80]
    movdqu xmm2, [rsp + 96]
    movdqu xmm3, [rsp + 112]
    movdqu xmm4, [rsp + 128]
    movdqu xmm5, [rsp + 144]
    movdqu xmm6, [rsp + 160]
    movdqu xmm7, [rsp + 176]
    leave
    jmp r11
    FUNCTION_END(import_thunk)

// Stand-in for functions we have not written yet: returns 0 (and 0.0).
    FUNCTION_BEGIN(import_return_zero)
    xor eax, eax
    xor edx, edx
    pxor xmm0, xmm0
    ret
    FUNCTION_END(import_return_zero)

#ifndef _WIN32
    .section .note.GNU-stack, "", @progbits
#endif
