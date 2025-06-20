pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl Perceptron {
    pub fn new(input_size: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; input_size],
            bias: 0.0,
            learning_rate,
        }
    }

    pub fn predict(&self, inputs: &[f64]) -> i32 {
        let sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;
        if sum >= 0.0 { 1 } else { 0 }
    }

    pub fn train(&mut self, inputs: &[f64], target: i32) {
        let prediction = self.predict(inputs);
        let error = (target - prediction) as f64;
        for (w, x) in self.weights.iter_mut().zip(inputs.iter()) {
            *w += self.learning_rate * error * x;
        }
        self.bias += self.learning_rate * error;
    }
}
