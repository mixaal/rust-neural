use rust_neural::{activation, builder::SequentialBuilder, layer::LayerType};

fn main() {
    let nn = SequentialBuilder::new()
        .add(LayerType::Input(3, Box::new(activation::Sigmoid)))
        .add(LayerType::Dense(5, Box::new(activation::ReLU)))
        .add(LayerType::Output(2, Box::new(activation::Softmax)))
        .build();
    let input = vec![0.5, 0.2, 0.1];
    let output = nn.forward(input);
    println!("Output: {:?}", output);
    println!("neural network: {:?}", nn);
    // Example output: Output: [0.1, 0.9]
}
