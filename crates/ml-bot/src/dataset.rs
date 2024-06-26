use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use arrayvec::ArrayVec;
use chess_ai_core::pgn::{IncrementalPgnParser, PgnGame};
use zstd::Decoder;

pub struct LichessDatasetIter<'a, const BATCH_SIZE: usize> {
    reader: BufReader<Decoder<'a, BufReader<File>>>,
    parser: IncrementalPgnParser,
}

pub struct LichessDataset<const BATCH_SIZE: usize> {
    path: PathBuf,
}

impl<const BATCH_SIZE: usize> LichessDataset<BATCH_SIZE> {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn new_iter(&self) -> LichessDatasetIter<'_, BATCH_SIZE> {
        LichessDatasetIter::open(self.path.as_path())
    }
}

impl<const BATCH_SIZE: usize> LichessDatasetIter<'_, BATCH_SIZE> {
    pub fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let decoder: Decoder<'_, BufReader<File>> = Decoder::new(file).unwrap();
        let reader = BufReader::new(decoder);
        
        Self {
            reader,
            parser: IncrementalPgnParser::default(),
        }
    }
}

impl<const BATCH_SIZE: usize> Iterator for LichessDatasetIter<'_, BATCH_SIZE> {
    type Item = [PgnGame; BATCH_SIZE];

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = ArrayVec::new();
        while batch.len() < BATCH_SIZE {
            let mut line = String::new();
            self.reader.read_line(&mut line).unwrap();
            let line = line.trim();
            self.parser.parse_line(line).unwrap();
            if self.parser.is_complete() {
                batch.push(self.parser.reset_and_convert());
            }
        }

        batch.into_inner().ok()
    }
}