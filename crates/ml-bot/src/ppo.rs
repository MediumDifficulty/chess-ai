use std::{fs::File, io::Write};

use arrayvec::ArrayVec;
use chess_ai_core::{
    game::{Board, GameOutcome, Move},
    movegen::{BoardCache, MoveGenDiagnostics},
    Engine,
};
use rand::Rng;
use random_bot::RandomEngine;
use tch::{
    kind::{FLOAT_CPU, FLOAT_CUDA, INT64_CPU, INT64_CUDA},
    nn::{self, OptimizerConfig},
    Device, Kind, Tensor,
};

use crate::{
    generate_output_map, get_highest_ranked_legal_move, get_inputs, BOARD_SIZE, BOARD_SQUARES,
    INPUT_SIZE, INT_INPUT_BITS, NON_SPATIAL_DATA_LEN, NUM_ACTIONS, TOTAL_PLANES,
};

const EPOCHS: usize = 1000;
const NSTEPS: i64 = 1 << INT_INPUT_BITS;
const OPT_BATCHSIZE: i64 = 32;
const OPT_EPOCHS: usize = 4;
const GAMES: i64 = 64;

const ENTROPY_COEFFICIENT: f64 = 0.01;
const DISCOUNT_FACTOR: f64 = 0.99;
const LEARNING_RATE: f64 = 1e-5;

const TRAIN_SIZE: i64 = NSTEPS * GAMES;

#[allow(clippy::type_complexity)]
pub fn model(p: &nn::Path) -> Box<dyn Fn(&Tensor) -> (Tensor, Tensor)> {
    let conv_config = nn::ConvConfig {
        stride: 1, // default
        padding: 1,
        padding_mode: nn::PaddingMode::Zeros, // TODO: I think this does what I think it does
        ..Default::default()
    };

    let seq = nn::seq()
        .add(nn::conv2d(p / "c1", TOTAL_PLANES, 32, 3, conv_config))
        .add_fn(Tensor::relu)
        .add(nn::conv2d(p / "c2", 32, 16, 3, conv_config))
        .add_fn(|xs| xs.relu().flat_view());

    // let fc = nn::seq()
    //     .add(nn::linear(
    //         p / "l1",
    //         16 * BOARD_SQUARES + NON_SPATIAL_DATA_LEN,
    //         512,
    //         Default::default(),
    //     ))
    //     .add_fn(Tensor::relu);

    const CONV_OUTPUT: i64 = 16 * BOARD_SQUARES + NON_SPATIAL_DATA_LEN;

    let critic = nn::linear(p / "cl", CONV_OUTPUT, 1, Default::default());
    let actor = nn::linear(p / "al", CONV_OUTPUT, NUM_ACTIONS, Default::default());
    let device = p.device();

    Box::new(move |xs| {
        let xs = xs.to_device(device);
        let xs: [Tensor; 2] = xs
            .split_with_sizes([TOTAL_PLANES * BOARD_SQUARES, NON_SPATIAL_DATA_LEN], 1)
            .try_into()
            .unwrap();

        let [xs, non_spatial] = xs;
        let xs = xs.view([-1, TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE]);
        // println!("{}", xs.get(0));
        let xs = xs.apply(&seq);
        let xs = Tensor::cat(&[xs, non_spatial], 1);
        // let xs = xs.apply(&fc);

        (xs.apply(&critic), xs.apply(&actor))
    })
}

pub fn train() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = model(&vs.root());
    let mut opt = nn::Adam::default()
        .eps(1e-4)
        .build(&vs, LEARNING_RATE)
        .unwrap();

    let output_map = generate_output_map();

    let mut total_episodes = 0.;

    let s_states = Tensor::zeros([NSTEPS + 1, GAMES, INPUT_SIZE], FLOAT_CUDA);

    for epoch in 0..EPOCHS {
        let mut env = Environment::new(vec![Box::new(RandomEngine)]);

        let mut sum_rewards = Tensor::zeros([GAMES], FLOAT_CUDA);
        let mut total_rewards = 0.;
        s_states.get(0).copy_(&env.observe().to_device(device)); // Slight variation from example

        let s_values = Tensor::zeros([NSTEPS, GAMES], FLOAT_CUDA);
        let s_rewards = Tensor::zeros([NSTEPS, GAMES], FLOAT_CUDA);
        let s_actions = Tensor::zeros([NSTEPS, GAMES], INT64_CUDA);
        let s_masks = Tensor::zeros([NSTEPS, GAMES], FLOAT_CUDA);

        let mut tracker = EnvironmentTracker::default();
        for s in 0..NSTEPS {
            let (critic, actor) = tch::no_grad(|| model(&s_states.get(s)));
            let probs = actor.softmax(-1, Kind::Float);
            let step = env.step(&probs, &output_map, Some(&mut tracker), device);

            let reward = step.reward.to_device(device);
            let action = step.action.to_device(device);
            let observation = step.observation.to_device(device);
            let is_done = step.is_done.to_device(device);

            sum_rewards += &reward;
            total_rewards += f64::try_from((&sum_rewards * &is_done).sum(Kind::Float)).unwrap();
            total_episodes += f64::try_from(is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32) - is_done;
            sum_rewards *= &masks;

            s_actions.get(s).f_copy_(&action).unwrap();
            s_values.get(s).f_copy_(&critic.squeeze_dim(-1)).unwrap();
            s_states.get(s + 1).f_copy_(&observation).unwrap(); // TODO: Double check this
            s_rewards.get(s).f_copy_(&reward).unwrap();
            s_masks.get(s).f_copy_(&masks).unwrap();
            // break;
        }
        // tracker.save_game_to_file("./game.uci.txt", 0);
        // break;
        let states = s_states.narrow(0, 0, NSTEPS).view([TRAIN_SIZE, INPUT_SIZE]);
        let returns = {
            let r = Tensor::zeros([NSTEPS + 1, GAMES], FLOAT_CUDA);
            let critic = tch::no_grad(|| model(&s_states.get(-1)).0);
            r.get(-1).copy_(&critic.view([GAMES]));
            for s in (0..NSTEPS).rev() {
                let r_s = s_rewards.get(s) + r.get(s + 1) * s_masks.get(s) * DISCOUNT_FACTOR;
                r.get(s).copy_(&r_s);
            }
            r.narrow(0, 0, NSTEPS).view([TRAIN_SIZE, 1])
        };
        let actions = s_actions.view([TRAIN_SIZE]);
        for _index in 0..OPT_EPOCHS {
            let batch_indices = Tensor::randint(TRAIN_SIZE, [OPT_BATCHSIZE], INT64_CUDA);
            let states = states.index_select(0, &batch_indices);
            let actions = actions.index_select(0, &batch_indices);
            let returns = returns.index_select(0, &batch_indices);
            let (critic, actor) = model(&states);
            let log_probs = actor.log_softmax(-1, Kind::Float);
            let probs = actor.softmax(-1, Kind::Float);
            let action_log_probs = {
                let index = actions.unsqueeze(-1).to_device(device);
                log_probs.gather(-1, &index, false).squeeze_dim(-1)
            };
            let dist_entropy = (-log_probs * probs)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);
            let advantages = returns.to_device(device) - critic;
            let value_loss = (&advantages * &advantages).mean(Kind::Float);
            let action_loss = (-advantages.detach() * action_log_probs).mean(Kind::Float);
            let loss = value_loss * 0.5 + action_loss - dist_entropy * ENTROPY_COEFFICIENT;
            opt.backward_step_clip(&loss, 0.5);
        }

        println!("{}", f64::try_from(s_rewards.sum(Kind::Float)).unwrap());
        if epoch % 50 == 0 {
            vs.save(format!("models/{}.safetensors", epoch)).unwrap();
        }
    }
}

#[derive(Default)]
struct Game {
    board: Board,
    outcome: Option<GameOutcome>,
}

struct Environment {
    engines: Vec<Box<dyn Engine>>,
    engine_selection: [usize; GAMES as usize],
    games: [Game; GAMES as usize],
}

struct Step {
    pub observation: Tensor,
    pub reward: Tensor,
    pub is_done: Tensor,
    pub action: Tensor,
}

struct EnvironmentTracker {
    moves: [ArrayVec<Move, { NSTEPS as usize * 2 }>; GAMES as usize],
}

impl EnvironmentTracker {
    pub fn save_game_to_file(&self, path: &str, game_idx: usize) {
        let content = self.moves[game_idx]
            .iter()
            .map(|m| m.to_string().trim().to_string())
            .collect::<Vec<_>>()
            .join("\n");

        File::create(path)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
    }
}

impl Default for EnvironmentTracker {
    fn default() -> Self {
        Self {
            moves: [(); GAMES as usize].map(|_| ArrayVec::new()),
        }
    }
}

impl Environment {
    pub fn new(engines: Vec<Box<dyn Engine>>) -> Self {
        let mut rng = rand::thread_rng();

        let mut env = Self {
            games: [0; 64].map(|_| Game::default()),
            engine_selection: [0; GAMES as usize].map(|_| rng.gen_range(0..engines.len())),
            engines,
        };

        // return env;

        // Randomly make some games play the first move so the bot doesn't always play as white
        for (i, game) in env.games.iter_mut().enumerate() {
            if rng.gen() {
                let engine = env.engines[env.engine_selection[i]].as_ref();
                let move_ = engine.best_move(&game.board);
                game.board.make_move(&move_);
            }
        }

        env
    }

    pub fn step(
        &mut self,
        probs: &Tensor,
        output_map: &[Move; NUM_ACTIONS as usize],
        mut tracker: Option<&mut EnvironmentTracker>,
        device: Device,
    ) -> Step {
        let rewards = Tensor::zeros([GAMES], FLOAT_CPU);
        let is_done = Tensor::zeros([GAMES], FLOAT_CPU);
        let observation = Tensor::zeros([GAMES, INPUT_SIZE], FLOAT_CPU);
        let action = Tensor::zeros([GAMES], INT64_CPU);
        for (i, game) in self.games.iter_mut().enumerate() {
            if game.outcome.is_some() {
                // is_done.get[i] = 1.;
                continue;
            }
            let legal_moves = game
                .board
                .generate_legal_moves(&mut MoveGenDiagnostics::default());
            let move_idx = get_highest_ranked_legal_move(
                &probs.get(i as i64),
                &legal_moves,
                output_map,
                &game.board,
            );
            game.board.make_move(&output_map[move_idx]);

            if let Some(ref mut tracker) = tracker {
                tracker.moves[i].push(output_map[move_idx].clone());
            }
            let _ = action.get(i as i64).fill_(move_idx as i64);

            let mut reward = 0.;
            let mut game_complete = false;

            // let legal_indices = Tensor::from_slice(
            //     &legal_moves
            //         .iter()
            //         .map(|m| output_map.iter().position(|e| e == m).unwrap() as i64)
            //         .collect::<Vec<_>>(),
            // ).to_device(device);
            // reward += f64::try_from(probs
            //     .get(i as i64)
            //     .index_select(0, &legal_indices)
            //     .sum(Kind::Float)).unwrap() / NSTEPS as f64;
            'a: {
                if let Some(outcome) = game.board.get_game_outcome(&mut BoardCache::default()) {
                    game.outcome = Some(outcome);
                    if outcome == GameOutcome::Checkmate {
                        reward += 1.;
                    }
                    game_complete = true;
                    break 'a;
                }

                // Opponent plays
                let engine = self.engines[self.engine_selection[i]].as_ref();
                let eng_move = engine.best_move(&game.board);
                game.board.make_move(&eng_move);

                if let Some(ref mut tracker) = tracker {
                    tracker.moves[i].push(eng_move);
                }

                if let Some(outcome) = game.board.get_game_outcome(&mut BoardCache::default()) {
                    game.outcome = Some(outcome);
                    if outcome == GameOutcome::Checkmate {
                        reward -= 1.;
                    }
                    game_complete = true;
                }
            }

            let _ = rewards.get(i as i64).fill_(reward);
            let _ = is_done
                .get(i as i64)
                .fill_(if game_complete { 1. } else { 0. });
            observation.get(i as i64).copy_(&get_inputs(&game.board));
        }

        Step {
            observation,
            reward: rewards,
            is_done,
            action,
        }
    }

    fn observe(&self) -> Tensor {
        let stack = Tensor::stack(
            &self
                .games
                .iter()
                .map(|g| get_inputs(&g.board))
                .collect::<Vec<_>>(),
            0,
        );
        stack
    }
}

type PPOModel = Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>;

pub struct MLEngine {
    model: PPOModel,
    output_map: [Move; NUM_ACTIONS as usize],
}

impl MLEngine {
    pub fn new(model: PPOModel) -> Self {
        let output_map = generate_output_map();
        Self { model, output_map }
    }
}

impl Engine for MLEngine {
    fn best_move(&self, board: &Board) -> Move {
        let inputs = get_inputs(board);
        let inputs = inputs.view([1, -1]);
        let (_, actor) = (self.model)(&inputs);
        let probs = actor.softmax(-1, Kind::Float);
        let probs = probs.view([NUM_ACTIONS]);
        let legal_moves = board.generate_legal_moves(&mut MoveGenDiagnostics::default());
        let mv = get_highest_ranked_legal_move(&probs, &legal_moves, &self.output_map, board);
        self.output_map[mv].clone()
    }
}
