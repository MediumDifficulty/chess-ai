use std::ops::Index;

use crate::game::{BoardPos, CoordOffsetTyp, CoordTyp};

pub fn add_board_offset(c: CoordTyp, o: CoordOffsetTyp) -> Option<CoordTyp> {
    c.checked_add_signed(o).filter(|&x| x < 8)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitBoard(pub u64);

impl BitBoard {
    fn index(pos: BoardPos) -> usize {
        (pos.rank * 8 + pos.file) as usize
    }

    pub fn set(&mut self, pos: BoardPos, value: bool) {
        self.0 |= (value as u64) << Self::index(pos);
    }

    pub fn get(&self, pos: BoardPos) -> bool {
        (self.0 >> Self::index(pos)) & 1 == 1
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

impl Index<BoardPos> for BitBoard {
    type Output = bool;

    fn index(&self, index: BoardPos) -> &Self::Output {
        if self.get(index) {
            &true
        } else {
            &false
        }
    }
}
