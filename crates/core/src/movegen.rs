use std::collections::HashMap;

use crate::{game::{Board, BoardPos, CoordOffsetTyp, Move, Piece}, util::BitBoard};

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct MoveGenDiagnostics {
    pub captures: usize,
    pub en_passant: usize,
    pub castles: usize,
    pub promotions: usize,
    pub checks: usize,
    pub discovered_checks: usize,
    pub double_checks: usize,
    pub checkmates: usize,
    pub sliding_moves: usize,
    pub pawn_moves: usize,
    pub offsetting_moves: usize,
}

impl MoveGenDiagnostics {
    pub fn total(&self) -> usize {
        self.captures
            + self.en_passant
            + self.castles
            + self.promotions
            + self.checks
            + self.discovered_checks
            + self.double_checks
            + self.checkmates
    }
}

impl std::ops::Add for MoveGenDiagnostics {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            captures: self.captures + rhs.captures,
            en_passant: self.en_passant + rhs.en_passant,
            castles: self.castles + rhs.castles,
            promotions: self.promotions + rhs.promotions,
            checks: self.checks + rhs.checks,
            discovered_checks: self.discovered_checks + rhs.discovered_checks,
            double_checks: self.double_checks + rhs.double_checks,
            checkmates: self.checkmates + rhs.checkmates,
            sliding_moves: self.sliding_moves + rhs.sliding_moves,
            pawn_moves: self.pawn_moves + rhs.pawn_moves,
            offsetting_moves: self.offsetting_moves + rhs.offsetting_moves,
        }
    }
}

impl std::ops::AddAssign for MoveGenDiagnostics {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

#[derive(Default)]
pub struct BoardCache {
    move_table: HashMap<Board, Vec<Move>>,
}

impl Board {
    fn generate_sliding_moves(
        &self,
        pos: BoardPos,
        diagnostics: &mut MoveGenDiagnostics,
    ) -> Vec<Move> {
        let piece = self[pos];
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

        let piece_offsets = if piece.is_rook() {
            &offsets[0..4]
        } else if piece.is_bishop() {
            &offsets[4..8]
        } else if piece.is_queen() {
            &offsets
        } else {
            // Piece is not sliding
            return vec![];
        };

        let mut moves = vec![];

        for &offset in piece_offsets {
            let mut current_pos = pos.add_offset(offset);

            while let Some(p) = current_pos {
                let piece_at_pos = self[p];

                if piece_at_pos.is_empty() {
                    moves.push(Move::new(pos, p));
                    current_pos = p.add_offset(offset);
                    continue;
                }

                if piece_at_pos.is_opponent(self.white_to_move()) {
                    moves.push(Move::new(pos, p));
                    diagnostics.captures += 1;
                }

                break;
            }
        }

        moves
    }

    fn generate_pawn_moves(
        &self,
        pos: BoardPos,
        diagnostics: &mut MoveGenDiagnostics,
    ) -> Vec<Move> {
        if !self[pos].is_pawn() {
            // Piece is not a pawn
            return vec![];
        }

        let rank_offset = self.white_to_move() as CoordOffsetTyp * 2 - 1;

        let mut destinations = vec![];

        // Add single square moves

        // Square in front of pawn
        if pos.add_offset((0, rank_offset)).map(|e| self[e]) == Some(Piece::Empty) {
            destinations.push(pos.add_offset((0, rank_offset)).unwrap());
        }

        // Take to both sides
        if pos
            .add_offset((1, rank_offset))
            .map(|e| self[e])
            .is_some_and(|p| p.is_opponent(self.white_to_move()))
        {
            destinations.push(pos.add_offset((1, rank_offset)).unwrap());
            diagnostics.captures += 1;
        }

        if pos
            .add_offset((-1, rank_offset))
            .map(|e| self[e])
            .is_some_and(|p| p.is_opponent(self.white_to_move()))
        {
            destinations.push(pos.add_offset((-1, rank_offset)).unwrap());
            diagnostics.captures += 1;
        }

        let mut moves = vec![];

        // Add promotions
        let promotion_rank = if self.white_to_move() { 6 } else { 1 };

        if pos.rank == promotion_rank {
            diagnostics.promotions += destinations.len() * 4;
            for &dest in &destinations {
                moves.push(Move::new_promotion(
                    pos,
                    dest,
                    Piece::WQueen.to_side(self.white_to_move()),
                ));
                moves.push(Move::new_promotion(
                    pos,
                    dest,
                    Piece::WRook.to_side(self.white_to_move()),
                ));
                moves.push(Move::new_promotion(
                    pos,
                    dest,
                    Piece::WBishop.to_side(self.white_to_move()),
                ));
                moves.push(Move::new_promotion(
                    pos,
                    dest,
                    Piece::WKnight.to_side(self.white_to_move()),
                ));
            }
        } else {
            for &dest in &destinations {
                moves.push(Move::new(pos, dest));
            }
        }

        // Add double square moves
        let starting_rank = if self.white_to_move() { 1 } else { 6 };
        let double_offset = pos.add_offset((0, rank_offset * 2));
        let single_offset = pos.add_offset((0, rank_offset));
        if pos.rank == starting_rank
            && double_offset.map(|e| self[e]) == Some(Piece::Empty)
            && single_offset.map(|e| self[e]) == Some(Piece::Empty)
        {
            moves.push(Move::new(pos, double_offset.unwrap()));
        }

        // Add en passant
        if let Some(ep_file) = self.en_passant() {
            let ep_rank = if self.white_to_move() { 4 } else { 3 };
            // u8::MAX will never be a valid file
            if
                (ep_file == pos.file + 1 || ep_file == pos.file.checked_sub(1).unwrap_or(u8::MAX)) // En Passant is possible
                && ep_rank == pos.rank 
            {
                moves.push(Move::new(pos, BoardPos::new(ep_file, pos.rank.checked_add_signed(rank_offset).unwrap())));
            }
        }

        moves
    }

    fn generate_offsetting_moves(
        &self,
        pos: BoardPos,
        diagnostics: &mut MoveGenDiagnostics,
    ) -> Vec<Move> {
        let piece = self[pos];
        if !piece.is_knight() && !piece.is_king() {
            // Piece is not king or knight
            return vec![];
        }

        let mut moves = vec![];
        for offset in if piece.is_knight() {
            [
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ]
        } else {
            [
                (1, 1),
                (1, 0),
                (1, -1),
                (0, -1),
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, 1),
            ]
        } {
            if let Some(dest) = pos.add_offset(offset) {
                if self[dest].is_empty() {
                    moves.push(Move::new(pos, dest));
                }

                if self[dest].is_opponent(self.white_to_move()) {
                    moves.push(Move::new(pos, dest));
                    diagnostics.captures += 1;
                }
            }
        }

        moves
    }

    fn generate_castling_moves(&self, pos: BoardPos) -> Vec<Move> {
        if !self[pos].is_king() {
            return vec![];
        }

        let mut s = self.clone();
        s.white_to_move = !s.white_to_move;
        let mut attacked_squares = BitBoard::default();
        for m in s.generate_moves(&mut MoveGenDiagnostics::default(), false) {
            attacked_squares.set(m.to_pos, true);
        }

        if attacked_squares[Self::CASTLE_FROM_POS.to_side(self.white_to_move)] {
            // King is in check
            return vec![];
        }

        let mut moves = vec![];
        
        let pawn_rank = if self.white_to_move { 1 } else { 6 };
        'king_side: {
            if self.castling().get(self.white_to_move, true) {    
                for i in (Self::CASTLE_FROM_POS.file + 1)..7 {
                    let pos = BoardPos::new(i, Self::CASTLE_FROM_POS.to_side(self.white_to_move).rank);
                    if !self[pos].is_empty() || attacked_squares[pos] {
                        break 'king_side;
                    }
                }

                for i in Self::CASTLE_FROM_POS.file..=7 {
                    let pos = BoardPos::new(i, pawn_rank);
                    if self[pos].is_pawn() && self[pos].is_opponent(self.white_to_move) {
                        break 'king_side;
                    }
                }

                moves.push(Move::new(pos, Self::CASTLE_POSITIONS[1].to_side(self.white_to_move)))
            }
        }

        'queen_side: {
            if self.castling().get(self.white_to_move, false) {
                if !self[BoardPos::new(1, 0).to_side(self.white_to_move)].is_empty() {
                    break 'queen_side;
                }
                
                for i in 2..Self::CASTLE_FROM_POS.file {
                    let pos = BoardPos::new(i, Self::CASTLE_FROM_POS.to_side(self.white_to_move).rank);
                    if !self[pos].is_empty() || attacked_squares[pos] {
                        break 'queen_side;
                    }
                }

                for i in 1..=Self::CASTLE_FROM_POS.file {
                    let pos = BoardPos::new(i, pawn_rank);
                    if self[pos].is_pawn() && self[pos].is_opponent(self.white_to_move) {
                        break 'queen_side;
                    }
                }

                moves.push(Move::new(pos, Self::CASTLE_POSITIONS[0].to_side(self.white_to_move)))
            }
        }

        moves
    }

    pub fn generate_moves_at(
        &self,
        pos: BoardPos,
        diagnostics: &mut MoveGenDiagnostics,
        consider_castling: bool
    ) -> Vec<Move> {
        if self[pos].is_empty() || self[pos].is_white() != self.white_to_move() {
            return vec![];
        }

        let mut sliding = self.generate_sliding_moves(pos, diagnostics);
        let pawn = self.generate_pawn_moves(pos, diagnostics);
        let offsetting = self.generate_offsetting_moves(pos, diagnostics);
        // println!("{:?} s: {} p: {}, k: {}", pos, moves.len(), pawn.len(), knight.len());

        diagnostics.sliding_moves += sliding.len();
        diagnostics.pawn_moves += pawn.len();
        diagnostics.offsetting_moves += offsetting.len();

        sliding.extend(pawn);
        sliding.extend(offsetting);

        if consider_castling {
            sliding.extend(self.generate_castling_moves(pos));
        }

        sliding
    }

    pub fn generate_moves(&self, diagnostics: &mut MoveGenDiagnostics, consider_castling: bool) -> Vec<Move> {
        let mut moves = vec![];

        for file in 0..8 {
            for rank in 0..8 {
                moves.extend(self.generate_moves_at(BoardPos::new(file, rank), diagnostics, consider_castling));
            }
        }

        moves
    }

    pub fn generate_legal_moves(&self, diagnostics: &mut MoveGenDiagnostics) -> Vec<Move> {
        let pseudo_legal_moves = self.generate_moves(diagnostics, true);
        let mut legal_moves = vec![];

        for m in pseudo_legal_moves {
            let mut board_clone = self.clone();
            board_clone.make_move(&m);
            let king_pos = board_clone
                .find_first(Piece::WKing.to_side(!board_clone.white_to_move()))
                .unwrap();

            let responses = board_clone.generate_moves(&mut MoveGenDiagnostics::default(), false);

            if responses.iter().any(|mv| mv.to_pos == king_pos) {
                // println!("e");
                diagnostics.checks += 1;
                continue;
            } else {
                legal_moves.push(m);
            }
        }

        legal_moves
    }

    pub fn generate_legal_moves_cached(
        &self,
        diagnostics: &mut MoveGenDiagnostics,
        cache: &mut BoardCache,
    ) -> Vec<Move> {
        if let Some(moves) = cache.move_table.get(self) {
            return moves.clone();
        }

        let moves = self.generate_legal_moves(diagnostics);
        cache.move_table.insert(self.clone(), moves.clone());
        moves
    }
}
