use std::fmt::Debug;

use ndarray::{Array1, Array2};
use rand::Rng;

use crate::activation::Activation;

#[derive(Debug)]
// Enum to specify layer types
pub enum LayerType {
    Input(usize, Box<dyn Activation>),
    Output(usize, Box<dyn Activation>),
    Dense(usize, Box<dyn Activation>),
    // Add more layer types here
}

// Layer trait
pub trait Layer {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32>;
    fn output_size(&self) -> usize;
    fn get_type(&self) -> LayerType;
}

impl Debug for dyn Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer")
            .field("type", &self.get_type())
            .finish()
    }
}

#[derive(Debug)]
// Input Layer
pub struct InputLayer {
    pub size: usize,
    pub activation: Box<dyn Activation>,
}

impl InputLayer {
    pub fn new(size: usize, output_size: usize, activation: Box<dyn Activation>) -> Self {
        assert_eq!(
            size, output_size,
            "Input size must match output size for InputLayer"
        );
        Self { size, activation }
    }
}

impl Layer for InputLayer {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.activation.activate(input)
    }

    fn output_size(&self) -> usize {
        self.size
    }

    fn get_type(&self) -> LayerType {
        LayerType::Input(self.size, self.activation.clone())
    }
}

// Dense Layer
pub struct DenseLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Box<dyn Activation>,
}
impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Box<dyn Activation>) -> Self {
        let mut rng = rand::rng();
        let weights =
            Array2::from_shape_fn((output_size, input_size), |_| rng.random_range(-0.5..0.5));
        let biases = Array1::zeros(output_size);
        Self {
            weights,
            biases,
            activation,
        }
    }
}
impl Layer for DenseLayer {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let z = self.weights.dot(input) + &self.biases;
        self.activation.activate(&z)
    }

    fn output_size(&self) -> usize {
        self.biases.len()
    }

    fn get_type(&self) -> LayerType {
        LayerType::Dense(self.output_size(), self.activation.clone())
    }
}

// Output Layer
pub struct OutputLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Box<dyn Activation>,
}
impl OutputLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Box<dyn Activation>) -> Self {
        let mut rng = rand::rng();
        let weights =
            Array2::from_shape_fn((output_size, input_size), |_| rng.random_range(-0.5..0.5));
        let biases = Array1::zeros(output_size);
        Self {
            weights,
            biases,
            activation,
        }
    }
}
impl Layer for OutputLayer {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let z = self.weights.dot(input) + &self.biases;
        self.activation.activate(&z)
    }

    fn output_size(&self) -> usize {
        self.biases.len()
    }

    fn get_type(&self) -> LayerType {
        LayerType::Output(self.output_size(), self.activation.clone())
    }
}
