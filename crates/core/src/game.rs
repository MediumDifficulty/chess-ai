use std::{collections::HashMap, fmt::{format, Display, Formatter}, ops::{Index, IndexMut}, str::FromStr};

use anyhow::{anyhow, Context, Result};
use derivative::Derivative;

use crate::{
    movegen::{BoardCache, MoveGenDiagnostics},
    util::BitBoard,
};

pub type CoordTyp = u8;
pub type CoordOffsetTyp = i8;

type BoardData = [[Piece; 8]; 8];

#[derive(Derivative)]
#[derivative(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Board {
    pieces: BoardData,
    pub white_to_move: bool,
    castling: CastlingRights,
    en_passant: Option<CoordTyp>,
    pub total_moves: usize,
    pub halfmoves: usize,
    #[derivative(Hash = "ignore")]
    pub position_counts: HashMap<BoardData, usize>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Piece {
    Empty,
    WKing,
    WQueen,
    WRook,
    WBishop,
    WKnight,
    WPawn,
    BKing,
    BQueen,
    BRook,
    BBishop,
    BKnight,
    BPawn,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum GameOutcome {
    Checkmate,
    Draw(DrawType),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DrawType {
    Stalemate,
    InsufficientMaterial,
    ThreefoldRepetition,
    FiftyMoveRule,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct CastlingRights {
    pub white_king: bool,
    pub white_queen: bool,
    pub black_king: bool,
    pub black_queen: bool,
}

impl CastlingRights {
    pub fn get(&self, white: bool, king_side: bool) -> bool {
        match (white, king_side) {
            (true, true) => self.white_king,
            (true, false) => self.white_queen,
            (false, true) => self.black_king,
            (false, false) => self.black_queen,
        }
    }

    pub fn invalidate(&mut self, white: bool, value: (bool, bool)) {
        if white {
            self.white_queen &= !value.0;
            self.white_king &= !value.1;
        } else {
            self.black_queen &= !value.0;
            self.black_king &= !value.1;
        }
    }

    pub fn to_fen(&self) -> String {
        if !(self.white_king || self.white_queen || self.black_king || self.white_queen) {
            return "-".to_string();
        }

        let p = [
            if self.white_king { Piece::WKing } else { Piece::Empty },
            if self.white_queen { Piece::WQueen } else { Piece::Empty },
            if self.black_king { Piece::BKing } else { Piece::Empty },
            if self.black_queen { Piece::BQueen } else { Piece::Empty }, 
        ].map(|p| format!("{p}"));

        p.join("")
    }
}

impl Board {
    pub const DEFAULT_FEN: &'static str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    pub const CASTLE_POSITIONS: [BoardPos; 2] = [BoardPos::new(2, 0), BoardPos::new(6, 0)];
    pub const CASTLE_FROM_POS: BoardPos = BoardPos::new(4, 0);

    pub fn from_fen(fen: &str) -> Result<Board> {
        let mut pieces = [[Piece::Empty; 8]; 8];
        let mut row = 7;
        let mut col = 0;

        let segments: [&str; 6] = fen
            .split_ascii_whitespace()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(); //.context("Invalid fen")?;
        let [placement, player_move, castling, et_passant, halfmove, fullmove] = segments;
        for c in placement.chars() {
            match c {
                '0'..='9' => {
                    col += c
                        .to_digit(10)
                        .context(format!("Failed to parse number {c}"))?
                        as usize
                }
                '/' => {
                    row -= 1;
                    col = 0;
                }
                _ => {
                    pieces[row][col] = match c {
                        'K' => Piece::WKing,
                        'k' => Piece::BKing,
                        'Q' => Piece::WQueen,
                        'q' => Piece::BQueen,
                        'R' => Piece::WRook,
                        'r' => Piece::BRook,
                        'B' => Piece::WBishop,
                        'b' => Piece::BBishop,
                        'N' => Piece::WKnight,
                        'n' => Piece::BKnight,
                        'P' => Piece::WPawn,
                        'p' => Piece::BPawn,
                        _ => Err(anyhow!("Invalid fen character: {c}"))?,
                    };
                    col += 1;
                }
            }
        }

        let white_to_move = player_move == "w";
        let castling = CastlingRights {
            white_king: castling.contains('K'),
            white_queen: castling.contains('Q'),
            black_king: castling.contains('k'),
            black_queen: castling.contains('q'),
        };
        let et_passant = if et_passant == "-" {
            None
        } else {
            Some(char_to_file(et_passant.chars().next().unwrap())?)
        };
        let halfmoves = halfmove.parse()?;
        let total_moves = fullmove.parse()?;

        let board = Self {
            pieces,
            white_to_move,
            castling,
            en_passant: et_passant,
            total_moves,
            halfmoves,
            position_counts: HashMap::new(),
        };

        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let position = (0..8).rev()
            .map(|rank| {
                let mut rank_str = String::new();

                let mut consecutive_empty = 0;
                for file in 0..8 {
                    let pos = BoardPos::new(file, rank);
                    let piece = self[pos];
    
                    if piece.is_empty() {
                        consecutive_empty += 1;
                    } else {
                        if consecutive_empty > 0 {
                            rank_str.push_str(&consecutive_empty.to_string());
                            consecutive_empty = 0;
                        }
    
                        rank_str.push_str(&format!("{piece}"))
                    }
                }

                if consecutive_empty > 0 {
                    rank_str.push_str(&consecutive_empty.to_string());
                }

                rank_str
            }).collect::<Vec<_>>().join("/");
        
        let slide_to_move = if self.white_to_move { "w" } else { "b" };
        let castling = self.castling.to_fen();
        let en_passant = self.en_passant.map_or("-".to_string(), |file| {
            let ep_rank = if self.white_to_move { 4 } else { 5 };
            format!("{ep_rank}{}", file_to_char(file))
        });

        format!("{position} {slide_to_move} {castling} {en_passant} {} {}", self.halfmoves, self.total_moves)
    }

    pub fn repetitions(&self) -> usize {
        *self.position_counts.get(&self.pieces).unwrap_or(&0)
    }

    pub fn get(&self, pos: BoardPos) -> Option<Piece> {
        if pos.is_valid() {
            return Some(self[pos]);
        }

        None
    }

    pub fn castling(&self) -> &CastlingRights {
        &self.castling
    }

    pub fn en_passant(&self) -> Option<CoordTyp> {
        self.en_passant
    }

    pub fn set_en_passant(&mut self, en_passant: Option<CoordTyp>) {
        self.en_passant = en_passant;
    }

    pub fn find_first(&self, piece: Piece) -> Option<BoardPos> {
        for file in 0..8 {
            for rank in 0..8 {
                let pos = BoardPos::new(file, rank);
                if self[pos] == piece {
                    return Some(BoardPos::new(file, rank));
                }
            }
        }
        None
    }

    fn castle(&mut self, m: &Move) -> bool {
        if self[m.from_pos].is_king()
            && m.from_pos == Self::CASTLE_FROM_POS.to_side(self.white_to_move)
        {
            let king_side = m.to_pos == Self::CASTLE_POSITIONS[1].to_side(self.white_to_move);

            if king_side && self.castling.get(self.white_to_move, true) {
                let new_rook_pos = BoardPos::new(5, 0).to_side(self.white_to_move);
                let old_rook_pos = BoardPos::new(7, 0).to_side(self.white_to_move);

                self[m.to_pos] = self[m.from_pos];
                self[m.from_pos] = Piece::Empty;
                self[new_rook_pos] = Piece::WRook.to_side(self.white_to_move);
                self[old_rook_pos] = Piece::Empty;

                return true;
            }

            let queen_side = m.to_pos == Self::CASTLE_POSITIONS[0].to_side(self.white_to_move);

            if queen_side && self.castling.get(self.white_to_move, false) {
                let new_rook_pos = BoardPos::new(3, 0).to_side(self.white_to_move);
                let old_rook_pos = BoardPos::new(0, 0).to_side(self.white_to_move);

                self[m.to_pos] = self[m.from_pos];
                self[m.from_pos] = Piece::Empty;
                self[new_rook_pos] = Piece::WRook.to_side(self.white_to_move);
                self[old_rook_pos] = Piece::Empty;

                return true;
            }
        }

        false
    }

    fn make_en_passant(&mut self, m: &Move) -> bool {
        if let Some(ep_file) = self.en_passant() {
            if self[m.from_pos].is_pawn() // Piece is a pawn
                && m.to_pos.file == ep_file // Piece is moving to the en passant file
                && m.from_pos.file != m.to_pos.file // Piece is not moving to the same file
                && self[m.to_pos].is_empty() // Destination square is empty
            {
                self[m.to_pos] = self[m.from_pos];
                self[m.from_pos] = Piece::Empty;
                self[BoardPos::new(ep_file, m.from_pos.rank)] = Piece::Empty;
                
                return true;
            }
        }

        false
    }

    pub fn make_move(&mut self, m: &Move) {
        assert!(!self[m.to_pos].is_king(), "King captured!\nBoard:\n{self}\nMove: {m}");

        if self.castle(m) {
            self.castling.invalidate(self.white_to_move, (true, true));
            self.white_to_move = !self.white_to_move;
            self.en_passant = None;
            self.total_moves += 1;
            self.halfmoves += 1;
            self.position_counts.clear();
            self.position_counts.insert(self.pieces, 1);
            return;
        }

        if self.make_en_passant(m) {
            self.white_to_move = !self.white_to_move;
            self.en_passant = None;
            self.total_moves += 1;
            self.halfmoves = 0;
            self.position_counts.clear();
            self.position_counts.insert(self.pieces, 1);
            return;
        }

        // There was a capture
        if self[m.to_pos] != Piece::Empty {
            self.halfmoves = 0;
            self.position_counts.clear();

            if m.to_pos == BoardPos::new(0, 0).to_side(!self.white_to_move) {
                self.castling.invalidate(!self.white_to_move, (true, false));
            } else if m.to_pos == BoardPos::new(7, 0).to_side(!self.white_to_move) {
                self.castling.invalidate(!self.white_to_move, (false, true));
            }
        }

        if m.from_pos == BoardPos::new(0, 0).to_side(self.white_to_move) {
            self.castling.invalidate(self.white_to_move, (true, false));
        }

        if m.from_pos == BoardPos::new(7, 0).to_side(self.white_to_move) {
            self.castling.invalidate(self.white_to_move, (false, true));
        }

        let piece = self[m.from_pos];

        if piece.is_king() {
            self.castling.invalidate(self.white_to_move, (true, true));
        }

        self[m.to_pos] = m.promotion.map(|p| p.to_side(self.white_to_move)).unwrap_or(piece);
        self[m.from_pos] = Piece::Empty;

        // En passant
        let pawn_starting_rank = if self.white_to_move { 1 } else { 6 };
        let double_pawn_move_rank = if self.white_to_move { 3 } else { 4 };

        if self[m.to_pos].is_pawn()
            && m.from_pos.rank == pawn_starting_rank
            && m.to_pos.rank == double_pawn_move_rank
        {
            self.en_passant = Some(m.to_pos.file);
        } else {
            self.en_passant = None;
        }

        let pos = self.position_counts.get_mut(&self.pieces);
        if let Some(p) = pos {
            *p += 1;
        } else {
            self.position_counts.insert(self.pieces, 1);
        }

        self.total_moves += 1;
        self.white_to_move = !self.white_to_move;
    }

    pub fn get_game_outcome(&self, cache: &mut BoardCache) -> Option<GameOutcome> {
        let moves = self.generate_legal_moves_cached(&mut MoveGenDiagnostics::default(), cache);
        let mut attacked_squares = BitBoard::default();
        let mut c = self.clone();
        c.white_to_move = !c.white_to_move;
        for m in c.generate_moves(&mut MoveGenDiagnostics::default(), false) {
            attacked_squares.set(m.to_pos, true);
        }
        
        let king_square = self
            .find_first(Piece::WKing.to_side(self.white_to_move))
            .unwrap();

        if moves.is_empty() {
            if attacked_squares.get(king_square) {
                return Some(GameOutcome::Checkmate);
            } else {
                return Some(GameOutcome::Draw(DrawType::Stalemate));
            }
        }

        if !self.is_sufficient_material() {
            return Some(GameOutcome::Draw(DrawType::InsufficientMaterial));
        }

        if self.halfmoves >= 50 {
            return Some(GameOutcome::Draw(DrawType::FiftyMoveRule));
        }

        if let Some(&count) = self.position_counts.get(&self.pieces) {
            if count >= 3 {
                return Some(GameOutcome::Draw(DrawType::ThreefoldRepetition));
            }
        }

        None
    }

    fn is_sufficient_material(&self) -> bool {
        #[derive(Default)]
        struct PieceCount {
            pawns: usize,
            rooks: usize,
            queens: usize,
            bishops: (usize, usize),
            knights: usize,
        }

        let mut counts = [PieceCount::default(), PieceCount::default()];

        for rank in 0..8 {
            for file in 0..8 {
                let pos = BoardPos::new(file, rank);
                let polarity = (file + rank) % 2 == 0;
                let piece = self[pos];
                
                let side = if piece.is_white() {
                    &mut counts[0]
                } else if piece.is_black() {
                    &mut counts[1]
                } else {
                    // Piece is empty
                    continue;
                };

                match piece.to_side(true) {
                    Piece::WKing => {},
                    Piece::WQueen => side.queens += 1,
                    Piece::WRook => side.rooks += 1,
                    Piece::WKnight => side.knights += 1,
                    Piece::WPawn => side.pawns += 1,
                    Piece::WBishop => if polarity {
                        side.bishops.0 += 1;
                    } else {
                        side.bishops.1 += 1;
                    },
                    _ => unreachable!()
                }
            }
        }

        
        counts.iter().any(|side| {
            let total_bishops = side.bishops.0 + side.bishops.1;
            
            (side.pawns >= 1 || side.rooks >= 1 || side.queens >= 1)
            || (side.knights >= 1 && total_bishops >= 1)
            || (side.knights >= 2)
            || (side.bishops.0 >= 1 && side.bishops.1 >= 1)
            // TODO: Piece against king
            // https://www.reddit.com/r/chess/comments/se89db/a_writeup_on_definitions_of_insufficient_material/
        })
    }

    /// Returns if white is the current player
    pub fn white_to_move(&self) -> bool {
        self.white_to_move
    }
}

impl Index<BoardPos> for Board {
    type Output = Piece;

    fn index(&self, index: BoardPos) -> &Self::Output {
        &self.pieces[index.rank as usize][index.file as usize]
    }
}

impl IndexMut<BoardPos> for Board {
    fn index_mut(&mut self, index: BoardPos) -> &mut Self::Output {
        &mut self.pieces[index.rank as usize][index.file as usize]
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen(Self::DEFAULT_FEN).unwrap()
    }
}

impl Piece {
    pub fn is_white(self) -> bool {
        matches!(
            self,
            Piece::WKing
                | Piece::WQueen
                | Piece::WRook
                | Piece::WBishop
                | Piece::WKnight
                | Piece::WPawn
        )
    }

    pub fn is_black(self) -> bool {
        matches!(
            self,
            Piece::BKing
                | Piece::BQueen
                | Piece::BRook
                | Piece::BBishop
                | Piece::BKnight
                | Piece::BPawn
        )
    }

    pub fn is_opponent(self, white: bool) -> bool {
        if white {
            return self.is_black();
        }

        self.is_white()
    }

    pub fn is_empty(&self) -> bool {
        *self == Piece::Empty
    }

    pub fn is_king(self) -> bool {
        matches!(self, Piece::WKing | Piece::BKing)
    }

    pub fn is_pawn(self) -> bool {
        matches!(self, Piece::WPawn | Piece::BPawn)
    }

    pub fn is_rook(self) -> bool {
        matches!(self, Piece::WRook | Piece::BRook)
    }

    pub fn is_bishop(self) -> bool {
        matches!(self, Piece::WBishop | Piece::BBishop)
    }

    pub fn is_queen(self) -> bool {
        matches!(self, Piece::WQueen | Piece::BQueen)
    }

    pub fn is_knight(self) -> bool {
        matches!(self, Piece::WKnight | Piece::BKnight)
    }

    pub fn is_playing(self, white: bool) -> bool {
        self.is_white() == white
    }

    pub fn to_side(self, white: bool) -> Self {
        if white != self.is_white() {
            return self.opposite();
        }

        self
    }

    pub fn opposite(self) -> Self {
        match self {
            Piece::WKing => Piece::BKing,
            Piece::WQueen => Piece::BQueen,
            Piece::WRook => Piece::BRook,
            Piece::WBishop => Piece::BBishop,
            Piece::WKnight => Piece::BKnight,
            Piece::WPawn => Piece::BPawn,
            Piece::BKing => Piece::WKing,
            Piece::BQueen => Piece::WQueen,
            Piece::BRook => Piece::WRook,
            Piece::BBishop => Piece::WBishop,
            Piece::BKnight => Piece::WKnight,
            Piece::BPawn => Piece::WPawn,
            _ => self,
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Piece::WKing => "K",
                Piece::WQueen => "Q",
                Piece::WRook => "R",
                Piece::WBishop => "B",
                Piece::WKnight => "N",
                Piece::WPawn => "P",
                Piece::BKing => "k",
                Piece::BQueen => "q",
                Piece::BRook => "r",
                Piece::BBishop => "b",
                Piece::BKnight => "n",
                Piece::BPawn => "p",
                Piece::Empty => " ",
            }
        )
    }
}

impl FromStr for Piece {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "K" => Piece::WKing,
            "Q" => Piece::WQueen,
            "R" => Piece::WRook,
            "B" => Piece::WBishop,
            "N" => Piece::WKnight,
            "P" => Piece::WPawn,
            "k" => Piece::BKing,
            "q" => Piece::BQueen,
            "r" => Piece::BRook,
            "b" => Piece::BBishop,
            "n" => Piece::BKnight,
            "p" => Piece::BPawn,
            "" => Piece::Empty,
            _ => return Err(anyhow!("Invalid piece: {s}")),
        })
    }
}

fn char_to_file(c: char) -> Result<CoordTyp> {
    match c {
        'a' => Ok(0),
        'b' => Ok(1),
        'c' => Ok(2),
        'd' => Ok(3),
        'e' => Ok(4),
        'f' => Ok(5),
        'g' => Ok(6),
        'h' => Ok(7),
        _ => Err(anyhow!("Invalid file: {c}")),
    }
}

fn file_to_char(f: CoordTyp) -> char {
    (b'a' + f) as char
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct BoardPos {
    pub file: CoordTyp,
    pub rank: CoordTyp,
}

impl BoardPos {
    pub const fn new(file: CoordTyp, rank: CoordTyp) -> Self {
        Self { file, rank }
    }

    pub const fn flip_vert(&self) -> Self {
        Self {
            file: self.file,
            rank: 7 - self.rank,
        }
    }

    pub const fn to_side(&self, white_to_move: bool) -> Self {
        if white_to_move {
            *self
        } else {
            self.flip_vert()
        }
    }

    pub const fn is_valid(&self) -> bool {
        self.file < 8 && self.rank < 8
    }

    pub fn add_offset(&self, offset: (CoordOffsetTyp, CoordOffsetTyp)) -> Option<Self> {
        let p = Self {
            file: self.file.checked_add_signed(offset.0)?,
            rank: self.rank.checked_add_signed(offset.1)?,
        };

        if !p.is_valid() {
            return None;
        }

        Some(p)
    }
}

impl Display for BoardPos {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (b'a' + self.file) as char, self.rank + 1)
    }
}

impl FromStr for BoardPos {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        if s.len() != 2 {
            return Err(anyhow!("Invalid board position: {s}"));
        }

        Ok(Self {
            file: char_to_file(s.chars().next().unwrap())?,
            rank: s.chars().nth(1).unwrap().to_digit(10).unwrap() as CoordTyp - 1,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Move {
    pub from_pos: BoardPos,
    pub to_pos: BoardPos,
    pub promotion: Option<Piece>,
}

impl Move {
    pub fn new(from_pos: BoardPos, to_pos: BoardPos) -> Self {
        Self {
            from_pos,
            to_pos,
            ..Default::default()
        }
    }

    pub fn new_promotion(from_pos: BoardPos, to_pos: BoardPos, promotion: Piece) -> Self {
        Self {
            from_pos,
            to_pos,
            promotion: Some(promotion),
        }
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.from_pos, self.to_pos, self.promotion.unwrap_or(Piece::Empty))?;

        Ok(())
    }
}


impl FromStr for Move {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let from_pos = BoardPos::from_str(&s[0..2])?;
        let to_pos = BoardPos::from_str(&s[2..4])?;

        let promotion = if s.len() == 5 {
            Some(Piece::from_str(&s[4..5])?)
        } else {
            None
        };

        Ok(Move {
            from_pos,
            to_pos,
            promotion
        })
    }
}