//! Module containing tensor types.
//!
//! Two main types of tensors are available.
//!
//! The first one, [`Tensor`](struct.Tensor.html),
//! is an _owned_ tensor that is backed by [`ndarray`](https://crates.io/crates/ndarray).
//! This kind of tensor is used to pass input data for the inference.
//!
//! The second one, [`OrtOwnedTensor`](struct.OrtOwnedTensor.html), is used
//! internally to pass to the ONNX Runtime inference execution to place
//! its output values. It is built using a [`OrtOwnedTensorExtractor`](struct.OrtOwnedTensorExtractor.html)
//! following the builder pattern.
//!
//! Once "extracted" from the runtime environment, this tensor will contain an
//! [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
//! containing _a view_ of the data. When going out of scope, this tensor will free the required
//! memory on the C side.
//!
//! **NOTE**: Tensors are not meant to be built directly. When performing inference,
//! the [`Session::run()`](../session/struct.Session.html#method.run) method takes
//! an `ndarray::Array` as input (taking ownership of it) and will convert it internally
//! to a [`Tensor`](struct.Tensor.html). After inference, a [`OrtOwnedTensor`](struct.OrtOwnedTensor.html)
//! will be returned by the method which can be derefed into its internal
//! [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).

pub mod ndarray_tensor;
pub mod ort_owned_tensor;
pub mod ort_tensor;

pub use ort_owned_tensor::{DynOrtTensor, OrtOwnedTensor};
pub use ort_tensor::OrtTensor;

use crate::tensor::ort_owned_tensor::TensorPointerHolder;
use crate::{error::call_ort, OrtError, Result};
use onnxruntime_sys::{self as sys, OnnxEnumInt};
use std::{convert::TryInto as _, ffi, fmt, ptr, rc, result, string};

// FIXME: Use https://docs.rs/bindgen/0.54.1/bindgen/struct.Builder.html#method.rustified_enum
// FIXME: Add tests to cover the commented out types
/// Enum mapping ONNX Runtime's supported tensor types
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum TensorElementDataType {
    /// 32-bit floating point, equivalent to Rust's `f32`
    Float = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as OnnxEnumInt,
    /// Unsigned 8-bit int, equivalent to Rust's `u8`
    Uint8 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 as OnnxEnumInt,
    /// Signed 8-bit int, equivalent to Rust's `i8`
    Int8 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 as OnnxEnumInt,
    /// Unsigned 16-bit int, equivalent to Rust's `u16`
    Uint16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 as OnnxEnumInt,
    /// Signed 16-bit int, equivalent to Rust's `i16`
    Int16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 as OnnxEnumInt,
    /// Signed 32-bit int, equivalent to Rust's `i32`
    Int32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 as OnnxEnumInt,
    /// Signed 64-bit int, equivalent to Rust's `i64`
    Int64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 as OnnxEnumInt,
    /// String, equivalent to Rust's `String`
    String = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING as OnnxEnumInt,
    // /// Boolean, equivalent to Rust's `bool`
    // Bool = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL as OnnxEnumInt,
    // /// 16-bit floating point, equivalent to Rust's `f16`
    // Float16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 as OnnxEnumInt,
    /// 64-bit floating point, equivalent to Rust's `f64`
    Double = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE as OnnxEnumInt,
    /// Unsigned 32-bit int, equivalent to Rust's `u32`
    Uint32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 as OnnxEnumInt,
    /// Unsigned 64-bit int, equivalent to Rust's `u64`
    Uint64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 as OnnxEnumInt,
    // /// Complex 64-bit floating point, equivalent to Rust's `???`
    // Complex64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 as OnnxEnumInt,
    // /// Complex 128-bit floating point, equivalent to Rust's `???`
    // Complex128 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 as OnnxEnumInt,
    // /// Brain 16-bit floating point
    // Bfloat16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as OnnxEnumInt,
}

impl Into<sys::ONNXTensorElementDataType> for TensorElementDataType {
    fn into(self) -> sys::ONNXTensorElementDataType {
        use TensorElementDataType::*;
        match self {
            Float => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Uint8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            Int8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            Uint16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            Int16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            Int32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Int64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            String => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            // Bool => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
            // }
            // Float16 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            // }
            Double => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            Uint32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            Uint64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            // Complex64 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
            // }
            // Complex128 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
            // }
            // Bfloat16 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
            // }
        }
    }
}

/// Trait used to map Rust types (for example `f32`) to ONNX types (for example `Float`)
pub trait TypeToTensorElementDataType {
    /// Return the ONNX type for a Rust type
    fn tensor_element_data_type() -> TensorElementDataType;

    /// If the type is `String`, returns `Some` with utf8 contents, else `None`.
    fn try_utf8_bytes(&self) -> Option<&[u8]>;
}

macro_rules! impl_prim_type_to_ort_trait {
    ($type_:ty, $variant:ident) => {
        impl TypeToTensorElementDataType for $type_ {
            fn tensor_element_data_type() -> TensorElementDataType {
                // unsafe { std::mem::transmute(TensorElementDataType::$variant) }
                TensorElementDataType::$variant
            }

            fn try_utf8_bytes(&self) -> Option<&[u8]> {
                None
            }
        }
    };
}

impl_prim_type_to_ort_trait!(f32, Float);
impl_prim_type_to_ort_trait!(u8, Uint8);
impl_prim_type_to_ort_trait!(i8, Int8);
impl_prim_type_to_ort_trait!(u16, Uint16);
impl_prim_type_to_ort_trait!(i16, Int16);
impl_prim_type_to_ort_trait!(i32, Int32);
impl_prim_type_to_ort_trait!(i64, Int64);
// impl_type_trait!(bool, Bool);
// impl_type_trait!(f16, Float16);
impl_prim_type_to_ort_trait!(f64, Double);
impl_prim_type_to_ort_trait!(u32, Uint32);
impl_prim_type_to_ort_trait!(u64, Uint64);
// impl_type_trait!(, Complex64);
// impl_type_trait!(, Complex128);
// impl_type_trait!(, Bfloat16);

/// Adapter for common Rust string types to Onnx strings.
///
/// It should be easy to use both `String` and `&str` as [TensorElementDataType::String] data, but
/// we can't define an automatic implementation for anything that implements `AsRef<str>` as it
/// would conflict with the implementations of [TypeToTensorElementDataType] for primitive numeric
/// types (which might implement `AsRef<str>` at some point in the future).
pub trait Utf8Data {
    /// Returns the utf8 contents.
    fn utf8_bytes(&self) -> &[u8];
}

impl Utf8Data for String {
    fn utf8_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> Utf8Data for &'a str {
    fn utf8_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<T: Utf8Data> TypeToTensorElementDataType for T {
    fn tensor_element_data_type() -> TensorElementDataType {
        TensorElementDataType::String
    }

    fn try_utf8_bytes(&self) -> Option<&[u8]> {
        Some(self.utf8_bytes())
    }
}

/// Trait used to map onnxruntime types to Rust types
pub trait TensorDataToType: Sized + Clone + fmt::Debug {
    /// The tensor element type that this type can extract from
    fn tensor_element_data_type() -> TensorElementDataType;

    /// Extract an `ArrayView` from the ort-owned tensor.
    fn extract_data<'t, D>(
        shape: D,
        tensor_element_len: usize,
        tensor_ptr: rc::Rc<TensorPointerHolder>,
    ) -> Result<TensorData<'t, Self, D>>
    where
        D: ndarray::Dimension;
}

/// Represents the possible ways tensor data can be accessed.
///
/// This should only be used internally.
#[derive(Debug)]
pub enum TensorData<'t, T, D>
where
    D: ndarray::Dimension,
{
    /// Data resides in ort's tensor, in which case the 't lifetime is what makes this valid.
    /// This is used for data types whose in-memory form from ort is compatible with Rust's, like
    /// primitive numeric types.
    TensorPtr {
        /// The pointer ort produced. Kept alive so that `array_view` is valid.
        ptr: rc::Rc<TensorPointerHolder>,
        /// A view into `ptr`
        array_view: ndarray::ArrayView<'t, T, D>,
    },
    /// String data is output differently by ort, and of course is also variable size, so it cannot
    /// use the same simple pointer representation.
    // Since 't outlives this struct, the 't lifetime is more than we need, but no harm done.
    Strings {
        /// Owned Strings copied out of ort's output
        strings: ndarray::Array<T, D>,
    },
}

/// Implements `OwnedTensorDataToType` for primitives, which can use `GetTensorMutableData`
macro_rules! impl_prim_type_from_ort_trait {
    ($type_:ty, $variant:ident) => {
        impl TensorDataToType for &$type_ {
            fn tensor_element_data_type() -> TensorElementDataType {
                TensorElementDataType::$variant
            }

            fn extract_data<'t, D>(
                shape: D,
                _tensor_element_len: usize,
                tensor_ptr: rc::Rc<TensorPointerHolder>,
            ) -> Result<TensorData<'t, Self, D>>
            where
                D: ndarray::Dimension,
            {
                extract_primitive_array(shape, tensor_ptr.tensor_ptr).map(|v| {
                    TensorData::TensorPtr {
                        ptr: tensor_ptr,
                        array_view: v,
                    }
                })
            }
        }

        impl TensorDataToType for $type_ {
            fn tensor_element_data_type() -> TensorElementDataType {
                TensorElementDataType::$variant
            }

            fn extract_data<'t, D>(
                shape: D,
                _tensor_element_len: usize,
                tensor_ptr: rc::Rc<TensorPointerHolder>,
            ) -> Result<TensorData<'t, Self, D>>
            where
                D: ndarray::Dimension,
            {
                extract_primitive_array(shape, tensor_ptr.tensor_ptr).map(|v| {
                    TensorData::TensorPtr {
                        ptr: tensor_ptr,
                        array_view: v,
                    }
                })
            }
        }
    };
}

/// Construct an [ndarray::ArrayView] over an Ort tensor.
///
/// Only to be used on types whose Rust in-memory representation matches Ort's (e.g. primitive
/// numeric types like u32).
fn extract_primitive_array<'t, D, T: TensorDataToType>(
    shape: D,
    tensor: *mut sys::OrtValue,
) -> Result<ndarray::ArrayView<'t, T, D>>
where
    D: ndarray::Dimension,
{
    // Get pointer to output tensor float values
    let mut output_array_ptr: *mut T = ptr::null_mut();
    let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
    let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
        output_array_ptr_ptr as *mut *mut std::ffi::c_void;
    unsafe {
        crate::error::call_ort(|ort| {
            ort.GetTensorMutableData.unwrap()(tensor, output_array_ptr_ptr_void)
        })
    }
    .map_err(OrtError::GetTensorMutableData)?;
    assert_ne!(output_array_ptr, ptr::null_mut());

    let array_view = unsafe { ndarray::ArrayView::from_shape_ptr(shape, output_array_ptr) };
    Ok(array_view)
}

impl_prim_type_from_ort_trait!(f32, Float);
impl_prim_type_from_ort_trait!(u8, Uint8);
impl_prim_type_from_ort_trait!(i8, Int8);
impl_prim_type_from_ort_trait!(u16, Uint16);
impl_prim_type_from_ort_trait!(i16, Int16);
impl_prim_type_from_ort_trait!(i32, Int32);
impl_prim_type_from_ort_trait!(i64, Int64);
impl_prim_type_from_ort_trait!(f64, Double);
impl_prim_type_from_ort_trait!(u32, Uint32);
impl_prim_type_from_ort_trait!(u64, Uint64);

impl TensorDataToType for String {
    fn tensor_element_data_type() -> TensorElementDataType {
        TensorElementDataType::String
    }

    fn extract_data<'t, D: ndarray::Dimension>(
        shape: D,
        tensor_element_len: usize,
        tensor_ptr: rc::Rc<TensorPointerHolder>,
    ) -> Result<TensorData<'t, Self, D>> {
        // Total length of string data, not including \0 suffix
        let mut total_length = 0_u64;
        unsafe {
            call_ort(|ort| {
                ort.GetStringTensorDataLength.unwrap()(tensor_ptr.tensor_ptr, &mut total_length)
            })
            .map_err(OrtError::GetStringTensorDataLength)?
        }

        // In the JNI impl of this, tensor_element_len was included in addition to total_length,
        // but that seems contrary to the docs of GetStringTensorDataLength, and those extra bytes
        // don't seem to be written to in practice either.
        // If the string data actually did go farther, it would panic below when using the offset
        // data to get slices for each string.
        let mut string_contents = vec![0_u8; total_length as usize];
        // one extra slot so that the total length can go in the last one, making all per-string
        // length calculations easy
        let mut offsets = vec![0_u64; tensor_element_len as usize + 1];

        unsafe {
            call_ort(|ort| {
                ort.GetStringTensorContent.unwrap()(
                    tensor_ptr.tensor_ptr,
                    string_contents.as_mut_ptr() as *mut ffi::c_void,
                    total_length,
                    offsets.as_mut_ptr(),
                    tensor_element_len as u64,
                )
            })
            .map_err(OrtError::GetStringTensorContent)?
        }

        // final offset = overall length so that per-string length calculations work for the last
        // string
        debug_assert_eq!(0, offsets[tensor_element_len]);
        offsets[tensor_element_len] = total_length;

        let strings = offsets
            // offsets has 1 extra offset past the end so that all windows work
            .windows(2)
            .map(|w| {
                let start: usize = w[0].try_into().expect("Offset didn't fit into usize");
                let next_start: usize = w[1].try_into().expect("Offset didn't fit into usize");

                let slice = &string_contents[start..next_start];
                String::from_utf8(slice.into())
            })
            .collect::<result::Result<Vec<String>, string::FromUtf8Error>>()
            .map_err(OrtError::StringFromUtf8Error)?;

        let array = ndarray::Array::from_shape_vec(shape, strings)
            .expect("Shape extracted from tensor didn't match tensor contents");

        Ok(TensorData::Strings { strings: array })
    }
}
