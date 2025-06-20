use std::fmt::Debug;

use dyn_clone::DynClone;
use ndarray::Array1;

use crate::Float;

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Softmax,
}

// Activation function trait
pub trait Activation: DynClone {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32>;
    fn derivative(&self, x: &Array1<f32>) -> Array1<f32>;
    fn get_type(&self) -> ActivationType;
}

dyn_clone::clone_trait_object!(Activation);

impl Debug for dyn Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Activation")
            .field("type", &self.get_type())
            .finish()
    }
}

#[derive(Debug, Clone)]
// Example activation functions
pub struct ReLU;
impl Activation for ReLU {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| v.max(0.0))
    }

    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    fn get_type(&self) -> ActivationType {
        ActivationType::ReLU
    }
}

#[derive(Debug, Clone)]
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        let sig = self.activate(x);
        sig.mapv(|s| s * (1.0 - s))
    }

    fn get_type(&self) -> ActivationType {
        ActivationType::Sigmoid
    }
}

#[derive(Debug, Clone)]
pub struct Softmax;
impl Activation for Softmax {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Array1<f32> = x.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        exp / sum
    }

    fn derivative(&self, _x: &Array1<f32>) -> Array1<f32> {
        // Not used directly in most cases
        unimplemented!()
    }

    fn get_type(&self) -> ActivationType {
        ActivationType::Softmax
    }
}

pub fn relu(x: Float) -> Float {
    x.max(0.0)
}

pub fn relu_derivative(x: Float) -> Float {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: Float) -> Float {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn tanh(x: Float) -> Float {
    x.tanh()
}

pub fn tanh_derivative(x: Float) -> Float {
    1.0 - x.tanh().powi(2)
}

pub fn leaky_relu(x: Float) -> Float {
    if x > 0.0 { x } else { 0.01 * x }
}

pub fn leaky_relu_derivative(x: Float) -> Float {
    if x > 0.0 { 1.0 } else { 0.01 }
}

pub fn softmax(xs: &[Float]) -> Vec<Float> {
    let max = xs.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
    let exps: Vec<Float> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: Float = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
