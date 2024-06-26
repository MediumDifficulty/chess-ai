use std::{collections::HashMap, path::Path};

use anyhow::Result;
use chess_ai_core::{game::{Board, Move}, pgn::PgnGame};
use tch::{nn::{self, OptimizerConfig}, Device, Kind, Tensor};

use crate::{dataset::LichessDataset, generate_output_map, get_inputs, BOARD_SIZE, BOARD_SQUARES, NON_SPATIAL_DATA_LEN, NUM_ACTIONS, TOTAL_PLANES};

const BATCH_SIZE: usize = 512;
const EPOCHS: usize = 10;
const MOVES_PER_GAME: usize = 64;

pub fn train() -> Result<()> {
    let device = Device::cuda_if_available();
    let dataset = LichessDataset::<BATCH_SIZE>::new(Path::new("D:\\downloads\\torrent\\lichess_db_standard_rated_2016-02.pgn.zst"));
    let vs = tch::nn::VarStore::new(device);
    let model = model(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;

    let output_map = generate_output_map()
        .into_iter()
        .enumerate()
        .map(|(i, mv)| (mv, i as i64))
        .collect::<HashMap<_, _>>();

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0;
        let mut total_moves: i64 = 0;
        for batch in dataset.new_iter() {
            let mut batch_loss = 0.0;
            let mut batch_moves: i64 = 0;
            // println!("b");
            let mut game_list = GameList::new();
            for m in 0..MOVES_PER_GAME {
                // println!("{m}");
                let inputs = game_list.get_inputs();
                // println!("Input: {:?}", inputs.size());
                let (mask, complete, moves) = game_list.step(&batch, m, &output_map);
                // println!("Mask: {mask}");
                let selected = inputs.index(&[Some(&mask), None]);
                // println!("{:?}", selected.size());
                let outputs = model(&selected);
                // println!("outputs {:?}", outputs);
                let loss = outputs.cross_entropy_for_logits(&moves.to_device(device));
                // println!("Loss {:?}", loss);

                opt.backward_step(&loss);

                batch_loss += f64::try_from(&loss).unwrap();
                total_loss += batch_loss;
                batch_moves += i64::try_from(mask.sum(Kind::Int64)).unwrap();
                total_moves += batch_moves;

                if complete {
                    break;
                }
            }

            println!("Batch complete Avg loss: {} moves: {total_moves}", batch_loss / batch_moves as f64);
            // if total_moves > 1_000_000 {
            //     break 'e;
            // }
        }

        println!("Epoch: {epoch} Avg loss: {}", total_loss / total_moves as f64)
    }

    Ok(())
}

#[allow(clippy::type_complexity)]
pub fn model(p: &nn::Path) -> Box<dyn Fn(&Tensor) -> Tensor> {
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

    let linear = nn::seq()
        .add(nn::linear(
            p / "l1",
            16 * BOARD_SQUARES + NON_SPATIAL_DATA_LEN,
            NUM_ACTIONS,
            Default::default(),
        ))
        .add_fn(Tensor::relu);

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
        xs.apply(&linear)
    })
}

fn get_move_indices(games: &[PgnGame]) -> Vec<usize> {
    let mut indices = Vec::new();
    for game in games {
        indices.push(game.moves.len());
    }

    indices
}

struct GameList {
    games: [(Board, bool); BATCH_SIZE],
}

impl GameList {
    pub fn new() -> Self {
        Self { games: (0..BATCH_SIZE).map(|_| (Board::default(), true)).collect::<Vec<_>>().try_into().unwrap() }
    }

    /// Returns true if all games are complete.
    pub fn step(&mut self, games: &[PgnGame], move_index: usize, output_map: &HashMap<Move, i64>) -> (Tensor, bool, Tensor) {
        let mut complete = true;
        let mask = Tensor::zeros(BATCH_SIZE as i64, (Kind::Bool, Device::Cpu));
        let mut moves = vec![];

        for (i, game) in self.games.iter_mut().enumerate() {
            let mv = games[i].moves.get(move_index);
            if let Some(mv) = mv {
                let move_ = game.0.make_pgn_move(mv);
                moves.push(*output_map.get(&move_).unwrap_or_else(|| panic!("Undefined move: {move_}")));
                let _ = mask.get(i as i64).fill_(1);
                complete = false;
            } else {
                game.1 = false;
            }

        }

        (mask, complete, Tensor::from_slice(&moves))
    }

    pub fn get_inputs(&self) -> Tensor {
        let game_inputs = self
            .games
            .iter()
            // .filter(|(_, e)| *e)
            .map(|(b, _)| get_inputs(b))
            .collect::<Vec<_>>();

        Tensor::stack(&game_inputs, 0)
    }
}