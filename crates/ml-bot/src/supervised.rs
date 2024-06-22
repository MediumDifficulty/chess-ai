use std::{fs::File, io::{BufRead, BufReader}, path::Path};

use zstd::Decoder;

pub struct LichessDataset {

}

impl LichessDataset {
    pub fn load(path: &Path) {
        let file = File::open(path).unwrap();
        let mut decoder = Decoder::new(file).unwrap();

        let reader = BufReader::new(&mut decoder);
        for line in reader.lines() {
            println!("{}", line.unwrap());
        }
    }
}