use std::fmt::Display;

use crate::game::{Board, BoardPos};

const RANK_SEPARATOR: &str = " +---+---+---+---+---+---+---+---+";

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..8).rev() {
            writeln!(f, "{RANK_SEPARATOR}")?;
            write!(f, " |")?;
            for file in 0..8 {
                let pos = BoardPos::new(file, rank);
                let piece = self[pos];
                // let p = (file + rank) % 2 == 0;
                // let a = if p {
                //     "W"
                // } else {
                //     "B"
                // };
                write!(f, " {piece} |")?;
            }
            writeln!(f, " {}", rank + 1)?;
        }

        writeln!(f, "{RANK_SEPARATOR}")?;
        writeln!(f, "   a   b   c   d   e   f   g   h")?;
        writeln!(f)?;
        writeln!(f, "FEN: {}", self.to_fen())?;

        Ok(())
    }
}
