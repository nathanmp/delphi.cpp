// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

#pragma once

// Phase 2 deliberately carries only the x86-64 AVX2 implementation.  The
// source is built once per ggml CPU backend variant and therefore inherits
// that variant's architecture flags.
#if defined(__x86_64__) && defined(__AVX2__)
#define IQK_IMPLEMENT
#endif

// This phase intentionally uses the donor's AVX2 kernels even when another
// backend variant is compiled with wider ISA flags.
#ifdef HAVE_FANCY_SIMD
#undef HAVE_FANCY_SIMD
#endif

#if defined(_MSC_VER)
#define IQK_ALWAYS_INLINE inline
#else
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif
