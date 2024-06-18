use std::{collections::HashSet, io::stdin};

use rand::Rng;

use chess_ai_core::{
    game::{Board, BoardPos, GameOutcome, Move, Piece},
    movegen::{BoardCache, MoveGenDiagnostics},
    uci, Engine,
};
use random_bot::RandomEngine;
use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    nn::{self, OptimizerConfig},
    Device, Kind, Tensor,
};

/// Number of possible chess moves
const NUM_ACTIONS: i64 = 1968;

fn main() {
    println!("Starting...");
    let mut buf = String::new();
    stdin().read_line(&mut buf).unwrap();
    if buf.trim() == "uci" {
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);
        vs.load("C:/Users/Thomas/Programming/rust/chess-ai/crates/ml-bot/models/950.safetensors")
            .expect("Invalid model file");
        let model = model(&vs.root());
        let engine = MLEngine::new(model);
        uci::start_uci("Lamp", &engine)
    } else {
        train();
    }
}

struct MLEngine {
    model: Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>,
    output_map: [Move; NUM_ACTIONS as usize],
}

impl MLEngine {
    fn new(model: Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>) -> Self {
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
        let mv = get_highest_ranked_legal_move(&probs, &legal_moves, &self.output_map);
        self.output_map[mv].clone()
    }
}

fn generate_output_map() -> [Move; NUM_ACTIONS as usize] {
    let mut moves = Vec::new();
    for rank in 0..8 {
        for file in 0..8 {
            let pos = BoardPos::new(file, rank);

            // Queen moves
            let offsets = [
                // Rook moves
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                // Bishop moves
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ];

            for offset in offsets {
                let mut current_pos = pos.add_offset(offset);

                while let Some(p) = current_pos {
                    moves.push(Move::new(pos, p));
                    current_pos = p.add_offset(offset);
                }
            }

            // Knight moves
            for offset in [
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ] {
                if let Some(p) = pos.add_offset(offset) {
                    moves.push(Move::new(pos, p));
                }
            }
            // King moves, castling and pawn moves are included in the queen's range of mobility
        }
    }

    // Promotions
    for file in 0..8 {
        // Iterate over ranks 1 and 7
        for (rank, offset) in [(6, 1), (1, -1)] {
            let pos = BoardPos::new(file, rank);
            // Iterate over pawn takes
            for offset in [(-1, offset), (0, offset), (1, offset)] {
                if let Some(p) = pos.add_offset(offset) {
                    moves.push(Move::new_promotion(pos, p, Piece::WBishop));
                    moves.push(Move::new_promotion(pos, p, Piece::WKnight));
                    moves.push(Move::new_promotion(pos, p, Piece::WQueen));
                    moves.push(Move::new_promotion(pos, p, Piece::WRook));
                }
            }
        }
    }

    moves.try_into().unwrap()
}

const BOARD_SIZE: i64 = 8;
const PIECE_PLANES: i64 = 13;
const ADDITIONAL_PLANES: i64 = 1;
const TOTAL_PLANES: i64 = PIECE_PLANES + ADDITIONAL_PLANES;
const BOARD_SQUARES: i64 = BOARD_SIZE * BOARD_SIZE;

const INT_INPUT_BITS: i64 = 6;
const NON_SPATIAL_DATA_LEN: i64 = INT_INPUT_BITS * 2;

const INPUT_SIZE: i64 = TOTAL_PLANES * BOARD_SQUARES + NON_SPATIAL_DATA_LEN;

#[allow(clippy::type_complexity)]
fn model(p: &nn::Path) -> Box<dyn Fn(&Tensor) -> (Tensor, Tensor)> {
    let conv_config = nn::ConvConfig {
        stride: 1, // default
        padding: 3,
        padding_mode: nn::PaddingMode::Zeros, // TODO: I think this does what I think it does
        ..Default::default()
    };

    let seq = nn::seq()
        .add(nn::conv2d(p / "c1", TOTAL_PLANES, 32, 7, conv_config))
        .add_fn(Tensor::relu)
        .add(nn::conv2d(p / "c2", 32, 16, 7, conv_config))
        .add_fn(|xs| xs.relu().flat_view());

    let fc = nn::seq()
        .add(nn::linear(
            p / "l1",
            16 * BOARD_SQUARES + NON_SPATIAL_DATA_LEN,
            512,
            Default::default(),
        ))
        .add_fn(Tensor::relu);

    let critic = nn::linear(p / "cl", 512, 1, Default::default());
    let actor = nn::linear(p / "al", 512, NUM_ACTIONS, Default::default());
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
        let xs = xs.apply(&fc);

        (xs.apply(&critic), xs.apply(&actor))
    })
}

fn num_to_binary_input(num: u64, size: i64) -> Tensor {
    Tensor::from_slice(
        &(0..size)
            .map(|i| ((num >> (size as u32 - i as u32)) & 1) as f32)
            .collect::<Vec<_>>(),
    )
}

fn get_inputs(board: &Board) -> Tensor {
    let mut layers = [[[0f32; BOARD_SIZE as usize]; BOARD_SIZE as usize]; TOTAL_PLANES as usize];

    // Add piece positions
    for file in 0..BOARD_SIZE as u8 {
        for rank in 0..BOARD_SIZE as u8 {
            let pos = BoardPos::new(file, rank);
            let piece = board[pos];
            layers[piece as usize][rank as usize][file as usize] = 1.;
        }
    }

    if let Some(en_passant) = board.en_passant() {
        layers[13][if board.white_to_move { 4 } else { 3 }][en_passant as usize] = 1.;
    }

    if board.castling().black_king {
        layers[13][7][6] = 1.;
    }

    if board.castling().black_queen {
        layers[13][7][2] = 1.;
    }

    if board.castling().white_king {
        layers[13][0][6] = 1.;
    }

    if board.castling().white_queen {
        layers[13][0][2] = 1.;
    }

    let spatial = Tensor::stack(&layers.map(|p| Tensor::from_slice2(&p)), 0);
    let non_spatial = Tensor::cat(
        &[
            num_to_binary_input(board.total_moves as u64, INT_INPUT_BITS),
            num_to_binary_input(board.halfmoves as u64, INT_INPUT_BITS),
        ],
        0,
    );

    let cat = Tensor::cat(&[spatial.view(-1), non_spatial], 0);
    debug_assert_eq!(cat.size(), vec![INPUT_SIZE]);
    cat
}

const EPOCHS: usize = 1000;
const NSTEPS: i64 = 1 << INT_INPUT_BITS;
const OPT_BATCHSIZE: i64 = 64;
const OPT_EPOCHS: usize = 4;
const GAMES: i64 = 32;

const ENTROPY_COEFFICIENT: f64 = 0.01;
const DISCOUNT_FACTOR: f64 = 0.98;
const LEARNING_RATE: f64 = 1e-5;

const TRAIN_SIZE: i64 = NSTEPS * GAMES;

fn train() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = model(&vs.root());
    let mut opt = nn::Adam::default().eps(1e-5).build(&vs, LEARNING_RATE).unwrap();

    let mut env = Environment::new(vec![Box::new(RandomEngine)]);
    let output_map = generate_output_map();

    let mut sum_rewards = Tensor::zeros([GAMES], FLOAT_CPU);
    let mut total_episodes = 0.;
    
    let s_states = Tensor::zeros([NSTEPS + 1, GAMES, INPUT_SIZE], FLOAT_CPU);
    
    for epoch in 0..EPOCHS {
        let mut total_rewards = 0.;
        env.reset();
        s_states.get(0).copy_(&env.observe()); // Slight variation from example

        let s_values = Tensor::zeros([NSTEPS, GAMES], FLOAT_CPU);
        let s_rewards = Tensor::zeros([NSTEPS, GAMES], FLOAT_CPU);
        let s_actions = Tensor::zeros([NSTEPS, GAMES], INT64_CPU);
        let s_masks = Tensor::zeros([NSTEPS, GAMES], FLOAT_CPU);

        for s in 0..NSTEPS {
            let (critic, actor) = tch::no_grad(|| model(&s_states.get(s)));
            let probs = actor.softmax(-1, Kind::Float);
            let step = env.step(&probs, &output_map, device);

            sum_rewards += &step.reward;
            total_rewards += f64::try_from((&sum_rewards * &step.is_done).sum(Kind::Float)).unwrap();
            total_episodes += f64::try_from(step.is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32) - step.is_done;
            sum_rewards *= &masks;

            s_actions.get(s).f_copy_(&step.action).unwrap();
            s_values.get(s).f_copy_(&critic.squeeze_dim(-1)).unwrap();
            s_states.get(s + 1).f_copy_(&step.observation).unwrap(); // TODO: Double check this
            s_rewards.get(s).f_copy_(&step.reward).unwrap();
            s_masks.get(s).f_copy_(&masks).unwrap();
            // break;
        }
        let states = s_states.narrow(0, 0, NSTEPS).view([TRAIN_SIZE, INPUT_SIZE]);
        let returns = {
            let r = Tensor::zeros([NSTEPS + 1, GAMES], FLOAT_CPU);
            let critic = tch::no_grad(|| model(&s_states.get(-1)).0);
            // println!("{:?}", critic.size());
            r.get(-1).copy_(&critic.view([GAMES]));
            for s in (0..NSTEPS).rev() {
                let r_s = s_rewards.get(s) + r.get(s + 1) * s_masks.get(s) * DISCOUNT_FACTOR;
                r.get(s).copy_(&r_s);
            }
            r.narrow(0, 0, NSTEPS).view([TRAIN_SIZE, 1])
        };
        let actions = s_actions.view([TRAIN_SIZE]);
        for _index in 0..OPT_EPOCHS {
            let batch_indices = Tensor::randint(TRAIN_SIZE, [OPT_BATCHSIZE], INT64_CPU);
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

            // Normalize advantages
            // let advantages = (&advantages - advantages.mean(Kind::Float)) / (advantages.std(true) + 1e-8);

            let value_loss = (&advantages * &advantages).mean(Kind::Float);
            let action_loss = (-advantages.detach() * action_log_probs).mean(Kind::Float);
            let loss = value_loss * 0.5 + action_loss - dist_entropy * ENTROPY_COEFFICIENT;
            opt.backward_step_clip(&loss, 0.5);
        }
        // break;
        println!("{}", total_rewards);
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

impl Environment {
    pub fn new(engines: Vec<Box<dyn Engine>>) -> Self {
        let mut env = Self {
            engines,
            games: Default::default(),
            engine_selection: Default::default(),
        };
        env.reset();

        // Randomly make some games play the first move so the bot doesn't always play as white
        let mut rng = rand::thread_rng();
        for (i, game) in env.games.iter_mut().enumerate() {
            if rng.gen() {
                let engine = env.engines[env.engine_selection[i]].as_ref();
                let move_ = engine.best_move(&game.board);
                game.board.make_move(&move_);
            }
        }

        env
    }

    pub fn reset(&mut self) {
        for Game { board, .. } in self.games.iter_mut() {
            *board = Board::default();
        }
        self.engine_selection = [0; GAMES as usize];
        let mut rng = rand::thread_rng();
        self.engine_selection = self
            .engine_selection
            .map(|_| rng.gen_range(0..self.engines.len()));
    }

    pub fn step(&mut self, probs: &Tensor, output_map: &[Move; NUM_ACTIONS as usize], device: Device) -> Step {
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
            let move_ =
                get_highest_ranked_legal_move(&probs.get(i as i64), &legal_moves, output_map);
            game.board.make_move(&output_map[move_]);
            let _ = action.get(i as i64).fill_(move_ as i64);

            let mut reward = 0.;
            let mut game_complete = false;

            let legal_indices = Tensor::from_slice(
                &legal_moves
                    .iter()
                    .map(|m| output_map.iter().position(|e| e == m).unwrap() as i64)
                    .collect::<Vec<_>>(),
            ).to_device(device);
            reward += f64::try_from(probs
                .get(i as i64)
                .index_select(0, &legal_indices)
                .sum(Kind::Float)).unwrap() / NSTEPS as f64;
            'a: {
                if let Some(outcome) = game.board.get_game_outcome(&mut BoardCache::default()) {
                    game.outcome = Some(outcome);
                    if outcome == GameOutcome::Checkmate {
                        // reward += 1.;
                    }
                    game_complete = true;
                    break 'a;
                }

                // Opponent plays
                let engine = self.engines[self.engine_selection[i]].as_ref();
                game.board.make_move(&engine.best_move(&game.board));

                if let Some(outcome) = game.board.get_game_outcome(&mut BoardCache::default()) {
                    game.outcome = Some(outcome);
                    if outcome == GameOutcome::Checkmate {
                        // reward -= 1.;
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

fn get_highest_ranked_legal_move(
    probs: &Tensor,
    legal_moves: &[Move],
    output_map: &[Move; NUM_ACTIONS as usize],
) -> usize {
    debug_assert_eq!(probs.size(), vec![NUM_ACTIONS]);

    let probs = Vec::<f32>::try_from(probs).expect("Incorrect Tensor type");
    let mut probs = probs.iter().copied().enumerate().collect::<Vec<_>>();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let legal_moves = legal_moves.iter().collect::<HashSet<_>>();

    for (i, _) in probs {
        if legal_moves.contains(&output_map[i]) {
            return i;
        }
    }

    panic!("No legal moves found");
}
