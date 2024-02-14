// use environments::{classic_control::CartPoleEnv, Env};
// use samples::Cli;
// use structopt::StructOpt;

// fn main() {
//     let cli: Cli = Cli::from_args();

//     let n_episodes: u128 = cli.n_episodes;
//     let max_steps: u128 = cli.max_steps;

//     let mut env = CartPoleEnv::new(max_steps);

//     for i in 0..1 {
//         let mut state = env.reset();
//         loop {
//             println!("{}", env.render());
//             let (state, reward, terminated) = match env.step(0) {
//                 Ok(s) => s,
//                 Err(_) => {
//                     panic!("Episode {} failed", i);
//                 }
//             };
//             println!("Reward: {}", reward);
//             println!("State: {:?}", state);

//             if terminated {
//                 break;
//             }
//         }
//     }
// }
fn main() {}
