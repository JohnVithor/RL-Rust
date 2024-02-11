use reinforcement_learning::{
    action_selection::{UniformEpsilonGreed, UpperConfidenceBound},
    agent::{
        expected_sarsa, qlearning, sarsa, DiscreteAgent, ElegibilityTracesAgent, OneStepAgent,
    },
};
use std::rc::Rc;
use structopt::StructOpt;

pub fn save_json(path: &str, data: serde_json::Value) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    serde_json::to_writer(&mut file, &data)?;
    Ok(())
}

pub fn moving_average(window: usize, vector: &[f64]) -> Vec<f64> {
    let mut aux: usize = 0;
    let mut result: Vec<f64> = vec![];
    while aux < vector.len() {
        let end: usize = if aux + window < vector.len() {
            aux + window
        } else {
            vector.len()
        };
        let slice: &[f64] = &vector[aux..end];
        let r: f64 = slice.iter().sum();
        result.push(r / window as f64);
        aux = end;
    }
    result
}

#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust")]
pub struct Cli {
    /// Should the env be stochastic
    #[structopt(long = "stochastic_env")]
    pub stochastic_env: bool,

    /// Change the env's map, if possible
    #[structopt(long = "map", default_value = "4x4")]
    pub map: String,

    /// Number of episodes for the training
    #[structopt(long = "n_episodes", short = "n", default_value = "100000")]
    pub n_episodes: u128,

    /// Maximum number of steps per episode
    #[structopt(long = "max_steps", default_value = "100")]
    pub max_steps: u128,

    /// Learning rate of the RL agent
    #[structopt(long = "learning_rate", default_value = "0.05")]
    pub learning_rate: f64,

    /// Initial value for the exploration ratio
    #[structopt(long = "initial_epsilon", default_value = "1.0")]
    pub initial_epsilon: f64,

    /// Value to determine percentage of episodes where exploration is possible;
    #[structopt(long = "exploration_time", default_value = "0.5")]
    pub exploration_time: f64,

    /// Final value for the exploration ratio
    #[structopt(long = "final_epsilon", default_value = "0.0")]
    pub final_epsilon: f64,

    /// Confidence level for the UCB action selection strategy
    #[structopt(long = "confidence_level", default_value = "0.5")]
    pub confidence_level: f64,

    /// Discont factor to be used on the temporal difference calculation
    #[structopt(long = "discount_factor", default_value = "0.95")]
    pub discount_factor: f64,

    /// Lambda factor to be used on the eligibility traces algorithms
    #[structopt(long = "lambda_factor", default_value = "0.5")]
    pub lambda_factor: f64,

    /// Moving average window to be used on the visualization of results
    #[structopt(long = "moving_average_window", default_value = "100")]
    pub moving_average_window: usize,
}

pub fn get_agents(args: Cli) -> (Vec<Box<dyn DiscreteAgent>>, Vec<&'static str>) {
    let greedy_sarsa_agent = OneStepAgent::new(
        Box::new(UniformEpsilonGreed::new(
            args.initial_epsilon,
            Rc::new(move |a| {
                a - args.initial_epsilon / (args.exploration_time * args.n_episodes as f64)
            }),
            args.final_epsilon,
        )),
        sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
    );

    let greedy_qlearning_agent = OneStepAgent::new(
        Box::new(UniformEpsilonGreed::new(
            args.initial_epsilon,
            Rc::new(move |a| {
                a - args.initial_epsilon / (args.exploration_time * args.n_episodes as f64)
            }),
            args.final_epsilon,
        )),
        qlearning,
        args.learning_rate,
        0.0,
        args.discount_factor,
    );

    let greedy_expected_sarsa_agent = OneStepAgent::new(
        Box::new(UniformEpsilonGreed::new(
            args.initial_epsilon,
            Rc::new(move |a| {
                a - args.initial_epsilon / (args.exploration_time * args.n_episodes as f64)
            }),
            args.final_epsilon,
        )),
        expected_sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
    );

    let ucb_sarsa_agent = OneStepAgent::new(
        Box::new(UpperConfidenceBound::new(args.confidence_level)),
        sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
    );

    let ucb_qlearning_agent = OneStepAgent::new(
        Box::new(UpperConfidenceBound::new(args.confidence_level)),
        qlearning,
        args.learning_rate,
        0.0,
        args.discount_factor,
    );

    let ucb_expected_sarsa_agent = OneStepAgent::new(
        Box::new(UpperConfidenceBound::new(args.confidence_level)),
        expected_sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
    );

    let greedy_sarsa_trace_agent: ElegibilityTracesAgent = ElegibilityTracesAgent::new(
        Box::new(UniformEpsilonGreed::new(
            args.initial_epsilon,
            Rc::new(move |a| {
                a - args.initial_epsilon / (args.exploration_time * args.n_episodes as f64)
            }),
            args.final_epsilon,
        )),
        sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
        args.lambda_factor,
    );

    let greedy_qlearning_trace_agent: ElegibilityTracesAgent = ElegibilityTracesAgent::new(
        Box::new(UniformEpsilonGreed::new(
            args.initial_epsilon,
            Rc::new(move |a| {
                a - args.initial_epsilon / (args.exploration_time * args.n_episodes as f64)
            }),
            args.final_epsilon,
        )),
        qlearning,
        args.learning_rate,
        0.0,
        args.discount_factor,
        args.lambda_factor,
    );

    let greedy_expected_sarsa_trace_agent: ElegibilityTracesAgent = ElegibilityTracesAgent::new(
        Box::new(UniformEpsilonGreed::new(
            args.initial_epsilon,
            Rc::new(move |a| {
                a - args.initial_epsilon / (args.exploration_time * args.n_episodes as f64)
            }),
            args.final_epsilon,
        )),
        expected_sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
        args.lambda_factor,
    );

    let ucb_sarsa_trace_agent: ElegibilityTracesAgent = ElegibilityTracesAgent::new(
        Box::new(UpperConfidenceBound::new(args.confidence_level)),
        sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
        args.lambda_factor,
    );

    let ucb_qlearning_trace_agent: ElegibilityTracesAgent = ElegibilityTracesAgent::new(
        Box::new(UpperConfidenceBound::new(args.confidence_level)),
        qlearning,
        args.learning_rate,
        0.0,
        args.discount_factor,
        args.lambda_factor,
    );

    let ucb_expected_sarsa_trace_agent: ElegibilityTracesAgent = ElegibilityTracesAgent::new(
        Box::new(UpperConfidenceBound::new(args.confidence_level)),
        expected_sarsa,
        args.learning_rate,
        0.0,
        args.discount_factor,
        args.lambda_factor,
    );
    let agents: Vec<Box<dyn DiscreteAgent>> = vec![
        Box::new(greedy_sarsa_agent),
        Box::new(greedy_qlearning_agent),
        Box::new(greedy_expected_sarsa_agent),
        Box::new(ucb_sarsa_agent),
        Box::new(ucb_qlearning_agent),
        Box::new(ucb_expected_sarsa_agent),
        Box::new(greedy_sarsa_trace_agent),
        Box::new(greedy_qlearning_trace_agent),
        Box::new(greedy_expected_sarsa_trace_agent),
        Box::new(ucb_sarsa_trace_agent),
        Box::new(ucb_qlearning_trace_agent),
        Box::new(ucb_expected_sarsa_trace_agent),
    ];
    let identifiers = vec![
        "ε-Greedy One-Step Sarsa",
        "ε-Greedy One-Step Qlearning",
        "ε-Greedy One-Step Expected Sarsa",
        "UCB One-Step Sarsa",
        "UCB One-Step Qlearning",
        "UCB One-Step Expected Sarsa",
        "ε-Greedy Trace Sarsa",
        "ε-Greedy Trace Qlearning",
        "ε-Greedy Trace Expected Sarsa",
        "UCB Trace Sarsa",
        "UCB Trace Qlearning",
        "UCB Trace Expected Sarsa",
    ];
    (agents, identifiers)
}
