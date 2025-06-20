use std::vec::Vec;

use crate::activation::Activation;
use crate::layer::DenseLayer;
use crate::layer::InputLayer;
use crate::layer::Layer;
use crate::layer::LayerType;
use crate::layer::OutputLayer;
use std::fmt::Debug;

pub struct SequentialBuilder {
    layers: Vec<Box<dyn Layer>>,
}

impl SequentialBuilder {
    pub fn new() -> Self {
        SequentialBuilder { layers: Vec::new() }
    }

    pub fn add_dense(mut self, neurons: usize, activation: Box<dyn Activation>) -> Self {
        self.layers.push(Box::new(DenseLayer::new(
            self.layers.last().map_or(0, |l| l.output_size()),
            neurons,
            activation,
        )));
        self
    }

    pub fn add_input(mut self, size: usize, activation: Box<dyn Activation>) -> Self {
        self.layers
            .push(Box::new(InputLayer::new(size, size, activation)));
        self
    }

    pub fn add_output(mut self, size: usize, activation: Box<dyn Activation>) -> Self {
        self.layers.push(Box::new(OutputLayer::new(
            self.layers.last().map_or(0, |l| l.output_size()),
            size,
            activation,
        )));
        self
    }

    pub fn add(self, layer_type: LayerType) -> Self {
        match layer_type {
            LayerType::Dense(neurons, activation) => self.add_dense(neurons, activation),
            LayerType::Input(size, activation) => self.add_input(size, activation),
            LayerType::Output(size, activation) => self.add_output(size, activation),
        }
    }

    pub fn build(self) -> Sequential {
        Sequential {
            layers: self.layers,
        }
    }
}
// Sequential neural network
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &self.layers)
            .finish()
    }
}

impl Sequential {
    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        use ndarray::Array1;
        for layer in &self.layers {
            let input_array = Array1::from(input);
            input = layer.forward(&input_array).to_vec();
        }
        input
    }
}
