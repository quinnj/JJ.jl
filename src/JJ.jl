module JJ

using SIMD, Mmap

@generated function compress(bm::Vec{32, Bool})
    decl = "declare i32 @llvm.x86.avx2.pmovmskb(<32 x i8>)"
    ir = """
        ; convert byte vector to word vector
        %a.i16 = bitcast <32 x i8> %0 to <16 x i16>
        ; psllw
        %b.i16 = shl <16 x i16> %a.i16, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
        %b.i8  = bitcast <16 x i16> %b.i16 to <32 x i8>
        ; pmovmskb
        %r.i32 = call i32 @llvm.x86.avx2.pmovmskb(<32 x i8> %b.i8)
        ret i32 %r.i32
    """
    quote
        Base.@_inline_meta
        Base.llvmcall($(decl, ir), UInt32, Tuple{NTuple{32, Base.VecElement{Bool}}}, bm.elts)
    end
end

compress(bm) = ccall("llvm.x86.avx2.pmovmskb", llvmcall, UInt32, (NTuple{32, Base.VecElement{Bool}},), bm.elts)
_mm256_cmpeq_epi8(a, b) = ccall("llvm.x86.avx2.vpcmpeqb", llvmcall, Vec{32, UInt8}, (NTuple{32, Base.VecElement{UInt8}}, NTuple{32, Base.VecElement{UInt8}}), a.elts, b.elts)
_mm256_shuffle_epi8(a, b) = ccall("llvm.x86.avx2.pshuf.b", llvmcall, Vec{32, UInt8}, (NTuple{32, Base.VecElement{UInt8}}, NTuple{32, Base.VecElement{UInt8}}), a.elts, b.elts)
_mm_clmulepi64_si128(a, b) = ccall("llvm.x86.pclmulqdq", llvmcall, Vec{16, UInt8}, (NTuple{16, Base.VecElement{UInt8}}, NTuple{16, Base.VecElement{UInt8}}, Int32), a.elts, b.elts, 0)

const DEFAULTMAXDEPTH = 1024
roundup_n(a, n) = (a + (n - 1)) & ~(n - 1)

function build_parsed_json(buf::Vector{UInt8})
    pj = ParsedJson()
    allocateCapacity!(pj, length(buf))
    res = find_structural_bits!(buf, pj)
    # if res
        # res = unified_machine(buf, pj)
    # end
    return res, pj
end

struct ScopeIndex
    start_of_scope::Int
    scope_type::UInt8
end

mutable struct ParsedJson
    depth::Int
    location::Int
    tape_length::Int
    current_type::UInt8
    current_val::UInt64
    depthindex::ScopeIndex
    bytecapacity::Int
    depthcapacity::Int
    tapecapacity::Int
    stringcapacity::Int
    current_loc::UInt32
    n_structural_indexes::UInt32
    structural_indexes::Vector{UInt32}
    tape::Vector{UInt64}
    containing_scope_offset::Vector{UInt32}
    ret_address::Vector{Ptr{Cvoid}}
    string_buf::Vector{UInt8}
    current_string_buf_loc::Int
    isvalid::Bool
    ParsedJson() = new(0, 0, 0, 0, 0, ScopeIndex(0, 0), 0, 0, 0, 0, 0, 0, UInt32[], UInt64[], UInt32[], Ptr{Cvoid}[], UInt8[], 0, false)
end

function init!(pj::ParsedJson)
    pj.current_string_buf_loc = pj.string_buf
    pj.current_loc = 0
    pj.isvalid = false
    return
end

function allocateCapacity!(pj, len, maxdepth=DEFAULTMAXDEPTH)
    isvalid = false
    bytecapacity = 0
    pj.n_structural_indexes = 0
    max_structures = roundup_n(len, 64) + 2 + 7
    pj.structural_indexes = Mmap.mmap(Vector{UInt32}, max_structures)
    localtapecapacity = roundup_n(len, 64)
    localstringcapacity = roundup_n(div(5 * len, 3) + 32, 64)
    pj.string_buf = Mmap.mmap(Vector{UInt8}, localstringcapacity)
    pj.tape = Mmap.mmap(Vector{UInt64}, localtapecapacity)
    pj.containing_scope_offset = zeros(UInt32, maxdepth)
    pj.ret_address = fill(Ptr{Cvoid}(0), maxdepth)
    pj.bytecapacity = len
    pj.depthcapacity = maxdepth
    pj.tapecapacity = localtapecapacity
    pj.stringcapacity = localstringcapacity
    return true
end

function find_structural_bits!(bytes::Vector{UInt8}, pj)
    buf = pointer(bytes)
    len = length(bytes)
    base_ptr = pj.structural_indexes
    base = 0
    prev_iter_ends_odd_backslash = UInt64(0)
    prev_iter_inside_quote = UInt64(0)
    prev_iter_ends_pseudo_pred = UInt64(1)
    structurals = UInt64(0)
    quote_mask = UInt64(0)
    lenminus64 = len < 64 ? 0 : len - 64
    idx = 0
    error_mask = UInt64(0)
    while idx < lenminus64
        ccall("llvm.prefetch", llvmcall, Cvoid, (Ptr{UInt8}, Int32, Int32, Int32), buf + idx + UInt32(128), Int32(0), Int32(3), Int32(1))
        input_lo = vload(Vec{32, UInt8}, buf + idx)
        input_hi = vload(Vec{32, UInt8}, buf + idx + 32)
        odd_ends, prev_iter_ends_odd_backslash = find_odd_backslash_sequences(input_lo, input_hi, prev_iter_ends_odd_backslash)
        quote_mask, prev_iter_inside_quote, quote_bits, error_mask = find_quote_mask_and_bits(input_lo, input_hi, odd_ends, prev_iter_inside_quote, error_mask)
        base = flatten_bits(base_ptr, base, idx, structurals)
        whitespace, structurals = find_whitespace_and_structurals(input_lo, input_hi, structurals)
        structurals, prev_iter_ends_pseudo_pred = finalize_structurals(structurals, whitespace, quote_mask, quote_bits, prev_iter_ends_pseudo_pred)
        idx += 64
    end

    if idx < len
        input_lo = vload(Vec{32, UInt8}, buf + idx)
        input_hi = vload(Vec{32, UInt8}, buf + idx + 32)
        odd_ends, prev_iter_ends_odd_backslash = find_odd_backslash_sequences(input_lo, input_hi, prev_iter_ends_odd_backslash)
        quote_mask, prev_iter_inside_quote, quote_bits, error_mask = find_quote_mask_and_bits(input_lo, input_hi, odd_ends, prev_iter_inside_quote, error_mask)
        base = flatten_bits(base_ptr, base, idx, structurals)
        whitespace, structurals = find_whitespace_and_structurals(input_lo, input_hi, structurals)
        structurals, prev_iter_ends_pseudo_pred = finalize_structurals(structurals, whitespace, quote_mask, quote_bits, prev_iter_ends_pseudo_pred)
        idx += 64
    end
    base = flatten_bits(base_ptr, base, idx, structurals)
    pj.n_structural_indexes = base
    if pj.n_structural_indexes == 0
        return false
    end
    if base_ptr[pj.n_structural_indexes] > len
        println("Internal bug\n")
        return false
    end
    if len != base_ptr[pj.n_structural_indexes]
        base_ptr[pj.n_structural_indexes] = len
        pj.n_structural_indexes += 1
    end
    base_ptr[pj.n_structural_indexes + 1] = 0
    return error_mask <= 0
end

function cmp_mask_against_input(input_lo, input_hi, mask)
    res_0 = compress(input_lo == mask)
    res_1 = compress(input_hi == mask)
    return UInt64(res_0) | (UInt64(res_1) << 32)
end

function find_odd_backslash_sequences(input_lo, input_hi, prev_iter_ends_odd_backslash)
    even_bits = 0x5555555555555555
    odd_bits = ~even_bits
    bs_bits = cmp_mask_against_input(input_lo, input_hi, Vec{32, UInt8}('\\' % UInt8))
    start_edges::UInt64 = bs_bits & ~(bs_bits << 1)
    even_start_mask::UInt64 = xor(even_bits, prev_iter_ends_odd_backslash)
    even_starts::UInt64 = start_edges & even_start_mask
    odd_starts::UInt64 = start_edges & ~even_start_mask
    even_carries::UInt64 = bs_bits + even_starts
    odd_carries::UInt64, iter_ends_odd_backslash = Base.add_with_overflow(bs_bits, odd_starts)
    odd_carries |= prev_iter_ends_odd_backslash
    prev_iter_ends_odd_backslash = iter_ends_odd_backslash ? UInt64(0x1) : UInt64(0)
    even_carry_ends = even_carries & ~bs_bits
    odd_carry_ends = odd_carries & ~bs_bits
    even_start_odd_end = even_carry_ends & odd_bits
    odd_start_even_end = odd_carry_ends & even_bits
    odd_ends = even_start_odd_end | odd_start_even_end
    return odd_ends, prev_iter_ends_odd_backslash
end

function unsigned_lteq_against_input(input_lo, input_hi, maxval)
    cmp_res_0 = max(maxval, input_lo) == maxval
    res_0::UInt64 = compress(cmp_res_0)
    cmp_res_1 = max(maxval, input_hi) == maxval
    res_1::UInt64 = compress(cmp_res_1)
    return res_0 | (res_1 << 32)
end

function find_quote_mask_and_bits(input_lo, input_hi, odd_ends, prev_iter_inside_quote, error_mask)
    quote_bits = cmp_mask_against_input(input_lo, input_hi, Vec{32, UInt8}('"' % UInt8))
    quote_bits = quote_bits & ~odd_ends
    quote_mask = reinterpret(Vec{2, UInt64}, 
                            _mm_clmulepi64_si128(
                                reinterpret(Vec{16, UInt8}, Vec{2, UInt64}((quote_bits, UInt64(0)))),
                                Vec{16, UInt8}(0xFF)
                            )
                         )[1]
    quote_mask = xor(quote_mask, prev_iter_inside_quote)
    unescaped::UInt64 = unsigned_lteq_against_input(input_lo, input_hi, Vec{32, UInt8}(0x1F))
    error_mask |= quote_mask & unescaped
    prev_iter_inside_quote = Core.bitcast(UInt64, Core.bitcast(Int64, quote_mask) >> 63)
    return quote_mask, prev_iter_inside_quote, quote_bits, error_mask
end

function flatten_bits(base_ptr, base, idx, bits)
    cnt::UInt32 = count_ones(bits);
    next_base::UInt32 = base + cnt;
    while bits != 0
        base_ptr[base + 1] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 2] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 3] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 4] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 5] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 6] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 7] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base_ptr[base + 8] = UInt32(idx - 64 + trailing_zeros(bits))
        bits = bits & (bits - 1)
        base += 8
    end
    return next_base
end

function find_whitespace_and_structurals(input_lo, input_hi, structurals)
    low_nibble_mask = Vec{32, UInt8}((16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0))
    high_nibble_mask = Vec{32, UInt8}((8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0))

    structural_shufti_mask = Vec{32, UInt8}(0x7)
    whitespace_shufti_mask = Vec{32, UInt8}(0x18)
    sevenf = Vec{32, UInt8}(0x7f)
    zeros = Vec{32, UInt8}(0)

    v_lo = _mm256_shuffle_epi8(low_nibble_mask, input_lo) &
                _mm256_shuffle_epi8(high_nibble_mask, (input_lo >> 4) & sevenf)

    v_hi = _mm256_shuffle_epi8(low_nibble_mask, input_hi) &
                _mm256_shuffle_epi8(high_nibble_mask, (input_hi >> 4) & sevenf)
    tmp_lo = (v_lo & structural_shufti_mask) == zeros
    tmp_hi = (v_hi & structural_shufti_mask) == zeros

    structural_res_0 = UInt64(compress(tmp_lo))
    structural_res_1 = UInt64(compress(tmp_hi))
    structurals = ~(structural_res_0 | (structural_res_1 << 32))

    tmp_ws_lo = (v_lo & whitespace_shufti_mask) == zeros
    tmp_ws_hi = (v_hi & whitespace_shufti_mask) == zeros

    ws_res_0 = UInt64(compress(tmp_ws_lo))
    ws_res_1 = UInt64(compress(tmp_ws_hi))
    whitespace = ~(ws_res_0 | (ws_res_1 << 32))
    return whitespace, structurals
end

function finalize_structurals(structurals, whitespace, quote_mask, quote_bits, prev_iter_ends_pseudo_pred)
    structurals &= ~quote_mask
    structurals |= quote_bits
    pseudo_pred = structurals | whitespace
    shifted_pseudo_pred = (pseudo_pred << 1) | prev_iter_ends_pseudo_pred
    prev_iter_ends_pseudo_pred = pseudo_pred >> 63
    pseudo_structurals = shifted_pseudo_pred & (~whitespace) & (~quote_mask)
    structurals |= pseudo_structurals
    structurals &= ~(quote_bits & ~quote_mask)
    return structurals, prev_iter_ends_pseudo_pred
end

function unified_machine(buf::Vector{UInt8}, len::Int, pj::ParsedJson)
    i = UInt32(0)
    idx = UInt32(0)
    c = UInt8(0)
    depth = UInt32(0)
    init!(pj)

end

end # module

########## load 64 bytes into a Vec{1, 64}
# _mm512_loadu_si512
# extern __m512i __cdecl _mm512_loadu_si512(void const* mem_addr);
# Load 512-bits of integer data from memory into destination.
# mem_addr does not need to be aligned on any particular boundary.

########## compare 64 byte vectors and compress result down to UInt64
# _mm512_cmpeq_epi8_mask
# __mmask64 _mm512_cmpeq_epi8_mask(__m512i a, __m512i b)
# CPUID Flags: AVX512BW
# Instruction(s): vpcmpb
# Compare packed 8-bit integers in a and b for equality, and and put each result in the corresponding bit of the returned mask value.
