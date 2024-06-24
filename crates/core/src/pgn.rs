
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, one_of, u64},
    combinator::{map, opt, value},
    multi::{many0, many1},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};

use crate::game::{BoardPos, Piece};

pub struct PgnGame {}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum TagType {
    // Official (https://en.wikipedia.org/wiki/Portable_Game_Notation)
    Event,
    Site,
    Date,
    Round,
    White,
    Black,
    Result,
    // Optional
    Annotator,
    PlyCount,
    TimeControl,
    Time,
    Termination,
    Mode,
    Fen,
    // Lichess
    UtcDate,
    UtcTime,
    WhiteElo,
    BlackElo,
    WhiteRatingDiff,
    BlackRatingDiff,
    Ec0,
    Opening,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct PgnTag {
    tag_type: TagType,
    value: String,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum PgnMoveBody {
    CastleKingSide,
    CastleQueenSide,
    PieceMove {
        piece: Piece,
        to: BoardPos,
        disambiguation: Disambiguation,
        promotion: Option<Piece>,
    },
}

#[derive(Debug)]
struct PgnMove {
    body: PgnMoveBody,
    comment: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Disambiguation {
    file: Option<u8>,
    rank: Option<u8>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum GameResult {
    Ongoing,
    Score { white: u64, black: u64 },
}

fn sp(i: &str) -> IResult<&str, &str> {
    let chars = " \t\r\n";

    take_while(move |c| chars.contains(c))(i)
}

fn string(input: &str) -> IResult<&str, &str> {
    delimited(char('"'), take_while(|c| c != '"'), char('"'))(input)
}

fn parse_tag_type(i: &str) -> IResult<&str, TagType> {
    alt((
        // Lichess
        alt((
            value(TagType::UtcDate, tag("UTCDate")),
            value(TagType::UtcTime, tag("UTCTime")),
            value(TagType::WhiteElo, tag("WhiteElo")),
            value(TagType::BlackElo, tag("BlackElo")),
            value(TagType::WhiteRatingDiff, tag("WhiteRatingDiff")),
            value(TagType::BlackRatingDiff, tag("BlackRatingDiff")),
            value(TagType::Ec0, tag("ECO")),
            value(TagType::Opening, tag("Opening")),
        )),
        value(TagType::Event, tag("Event")),
        value(TagType::Site, tag("Site")),
        value(TagType::Date, tag("Date")),
        value(TagType::Round, tag("Round")),
        // Make sure to check "White" and "Black" after "WhiteElo" and "BlackElo"
        value(TagType::White, tag("White")),
        value(TagType::Black, tag("Black")),
        value(TagType::Result, tag("Result")),
        value(TagType::Annotator, tag("Annotator")),
        value(TagType::PlyCount, tag("PlyCount")),
        value(TagType::TimeControl, tag("TimeControl")),
        value(TagType::Time, tag("Time")),
        value(TagType::Termination, tag("Termination")),
        value(TagType::Mode, tag("Mode")),
        value(TagType::Fen, tag("FEN")),
    ))(i)
}

fn parse_tag(i: &str) -> IResult<&str, PgnTag> {
    let t = delimited(
        preceded(char('['), sp),
        pair(parse_tag_type, preceded(sp, string)),
        preceded(sp, char(']')),
    )(i)?;

    Ok((
        t.0,
        PgnTag {
            tag_type: t.1 .0,
            value: t.1 .1.to_string(),
        },
    ))
}

fn move_number(i: &str) -> IResult<&str, u64> {
    let a = pair(u64, alt((tag("..."), tag("."))))(i)?;

    Ok((a.0, a.1 .0))
}

fn comment(i: &str) -> IResult<&str, String> {
    map(
        delimited(char('{'), take_while(|c| c != '}'), char('}')),
        |s: &str| s.to_string(),
    )(i)
}

fn parse_result(i: &str) -> IResult<&str, GameResult> {
    alt((
        value(GameResult::Ongoing, char('*')),
        map(pair(u64, preceded(char('-'), u64)), |s| GameResult::Score {
            white: s.0,
            black: s.1,
        }),
    ))(i)
}

fn file(i: &str) -> IResult<&str, u8> {
    alt((
        value(0, tag("a")),
        value(1, tag("b")),
        value(2, tag("c")),
        value(3, tag("d")),
        value(4, tag("e")),
        value(5, tag("f")),
        value(6, tag("g")),
        value(7, tag("h")),
    ))(i)
}

fn rank(i: &str) -> IResult<&str, u8> {
    // Yes, I know this is horrible
    alt((
        value(0, tag("1")),
        value(1, tag("2")),
        value(2, tag("3")),
        value(3, tag("4")),
        value(4, tag("5")),
        value(5, tag("6")),
        value(6, tag("7")),
        value(7, tag("8")),
    ))(i)
}

fn piece(i: &str) -> IResult<&str, Piece> {
    alt((
        value(Piece::WBishop, tag("B")),
        value(Piece::WKing, tag("K")),
        value(Piece::WKnight, tag("N")),
        value(Piece::WPawn, tag("P")),
        value(Piece::WQueen, tag("Q")),
        value(Piece::WRook, tag("R")),
    ))(i)
}

fn board_pos(i: &str) -> IResult<&str, BoardPos> {
    map(pair(file, rank), |f| BoardPos {
        file: f.0,
        rank: f.1,
    })(i)
}

fn parse_disambiguation(i: &str) -> IResult<&str, Disambiguation> {
    map(pair(opt(file), opt(rank)), |f| Disambiguation {
        file: f.0,
        rank: f.1,
    })(i)
}

fn check_indicator(i: &str) -> IResult<&str, char> {
    one_of("+#")(i)
}

fn parse_move(i: &str) -> IResult<&str, PgnMoveBody> {
    alt((
        value(PgnMoveBody::CastleQueenSide, tag("O-O-O")),
        value(PgnMoveBody::CastleKingSide, tag("O-O")),
        // I'm sure there's a way to tidy this up with less repetition...
        map(
            tuple((
                opt(piece),
                parse_disambiguation,
                opt(tag("x")), // Capture
                board_pos,
                opt(preceded(tag("="), piece)),
                opt(check_indicator),
            )),
            |(p, d, _, dest, promotion, _)| PgnMoveBody::PieceMove {
                piece: p.unwrap_or(Piece::WPawn),
                to: dest,
                disambiguation: d,
                promotion,
            },
        ),
        map(
            tuple((
                opt(piece),
                opt(tag("x")), // Capture
                board_pos,
                opt(preceded(tag("="), piece)),
                opt(check_indicator),
            )),
            |(p, _, dest, promotion, _)| PgnMoveBody::PieceMove {
                piece: p.unwrap_or(Piece::WPawn),
                to: dest,
                disambiguation: Disambiguation {
                    file: None,
                    rank: None,
                },
                promotion,
            },
        ),
    ))(i)
}

fn tokenise_moves(i: &str) -> IResult<&str, Vec<PgnMove>> {
    many0(preceded(
        pair(opt(move_number), sp),
        map(
            pair(parse_move, delimited(sp, opt(comment), sp)),
            |(mv, comment)| PgnMove { body: mv, comment },
        ),
    ))(i)
}

pub fn parse_pgn(src: &str) -> IResult<&str, PgnGame> {
    let (r, (tags, moves, result)) = tuple((
        many1(preceded(sp, parse_tag)),
        preceded(sp, tokenise_moves),
        preceded(sp, parse_result),
    ))(src)?;

    println!("Tags: {tags:?}");
    println!("Moves: {moves:#?}");
    println!("Result: {result:?}");

    todo!()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::{
        game::{BoardPos, Piece},
        pgn::{Disambiguation, PgnMoveBody},
    };

    #[test]
    fn test_parse_tag() {
        let src = "[WhiteElo \"1127\"]";
        let res = super::parse_tag(src);
        assert_eq!(
            res,
            Ok((
                "",
                super::PgnTag {
                    tag_type: super::TagType::WhiteElo,
                    value: "1127".to_string()
                }
            ))
        )
    }

    #[test]
    fn parse_move_number() {
        let src = "1... e4";
        let res = super::move_number(src);
        assert_eq!(res, Ok((" e4", 1)));
    }

    #[test]
    fn parse_board_pos() {
        let src = "a1";
        let res = super::board_pos(src);
        assert_eq!(res, Ok(("", BoardPos { file: 0, rank: 0 })));
    }

    #[test]
    fn parse_disambiguation() {
        let src = "a1";
        let res = super::parse_disambiguation(src);
        assert_eq!(
            res,
            Ok((
                "",
                Disambiguation {
                    file: Some(0),
                    rank: Some(0)
                }
            ))
        );

        let src = "a";
        let res = super::parse_disambiguation(src);
        assert_eq!(
            res,
            Ok((
                "",
                Disambiguation {
                    file: Some(0),
                    rank: None
                }
            ))
        );

        let src = "1";
        let res = super::parse_disambiguation(src);
        assert_eq!(
            res,
            Ok((
                "",
                Disambiguation {
                    file: None,
                    rank: Some(0)
                }
            ))
        );

        let src = "";
        let res = super::parse_disambiguation(src);
        assert_eq!(
            res,
            Ok((
                "",
                Disambiguation {
                    file: None,
                    rank: None
                }
            ))
        );
    }

    #[test]
    fn test_parse_move() {
        let src = "N6xf3=Q#";
        let res = super::parse_move(src);
        assert_eq!(
            res,
            Ok((
                "",
                PgnMoveBody::PieceMove {
                    piece: Piece::WKnight,
                    to: BoardPos::from_str("f3").unwrap(),
                    disambiguation: Disambiguation {
                        file: None,
                        rank: Some(5)
                    },
                    promotion: Some(Piece::WQueen)
                }
            ))
        );
    }

    #[test]
    fn test_parse_pgn() {
        let src = "[Event \"Rated Blitz game\"]
[Site \"https://lichess.org/Oy5s7AIm\"]
[Date \"2024.05.01\"]
[Round \"-\"]
[White \"zbluered\"]
[Black \"ttch\"]
[Result \"1-0\"]
[UTCDate \"2024.05.01\"]
[UTCTime \"00:10:53\"]
[WhiteElo \"2107\"]
[BlackElo \"2107\"]
[WhiteRatingDiff \"+6\"]
[BlackRatingDiff \"-6\"]
[ECO \"D00\"]
[Opening \"Queen's Pawn Game: Chigorin Variation, Irish Gambit\"]
[TimeControl \"180+0\"]
[Termination \"Time forfeit\"]

1. Nc3 { [%clk 0:03:00] } 1... d5 { [%clk 0:03:00] } 2. d4 { [%clk 0:02:59] } 2... c5 { [%clk 0:02:57] } 3. dxc5 { [%clk 0:02:58] } 3... Nf6 { [%clk 0:02:52] } 4. e3 { [%clk 0:02:49] } 4... e5 { [%clk 0:02:49] } 5. Nf3 { [%clk 0:02:42] } 5... Nc6 { [%clk 0:02:45] } 6. Bb5 { [%clk 0:02:40] } 6... Bg4 { [%clk 0:02:42] } 7. h3 { [%clk 0:02:38] } 7... Bh5 { [%clk 0:02:36] } 8. g4 { [%clk 0:02:36] } 8... Bg6 { [%clk 0:02:34] } 9. g5 { [%clk 0:02:35] } 9... Ne4 { [%clk 0:02:27] } 10. Nxe5 { [%clk 0:02:31] } 10... Bxc5 { [%clk 0:02:19] } 11. Nxc6 { [%clk 0:02:25] } 11... bxc6 { [%clk 0:02:02] } 12. Bxc6+ { [%clk 0:02:23] } 12... Kf8 { [%clk 0:02:01] } 13. Bxa8 { [%clk 0:02:22] } 13... Qxa8 { [%clk 0:01:59] } 14. Qxd5 { [%clk 0:02:21] } 14... Qxd5 { [%clk 0:01:56] } 15. Nxd5 { [%clk 0:02:20] } 15... h6 { [%clk 0:01:46] } 16. gxh6 { [%clk 0:02:18] } 16... Rxh6 { [%clk 0:01:42] } 17. Nf4 { [%clk 0:02:17] } 17... Bf5 { [%clk 0:01:40] } 18. Bd2 { [%clk 0:02:15] } 18... g5 { [%clk 0:01:36] } 19. Nd3 { [%clk 0:02:10] } 19... Bb6 { [%clk 0:01:33] } 20. Bb4+ { [%clk 0:02:07] } 20... Kg7 { [%clk 0:01:30] } 21. O-O-O { [%clk 0:02:00] } 21... f6 { [%clk 0:01:26] } 22. Kb1 { [%clk 0:01:57] } 22... Bxh3 { [%clk 0:01:20] } 23. f3 { [%clk 0:01:55] } 23... Ng3 { [%clk 0:01:16] } 24. Rhe1 { [%clk 0:01:50] } 24... Nf5 { [%clk 0:01:11] } 25. Bc5 { [%clk 0:01:45] } 25... Bg2 { [%clk 0:01:04] } 26. Bxb6 { [%clk 0:01:43] } 26... axb6 { [%clk 0:01:02] } 27. f4 { [%clk 0:01:42] } 27... g4 { [%clk 0:01:00] } 28. e4 { [%clk 0:01:36] } 28... Nd4 { [%clk 0:00:57] } 29. e5 { [%clk 0:01:33] } 29... fxe5 { [%clk 0:00:43] } 30. fxe5 { [%clk 0:01:31] } 30... Ne6 { [%clk 0:00:42] } 31. a4 { [%clk 0:01:21] } 31... Bb7 { [%clk 0:00:34] } 32. Rg1 { [%clk 0:01:18] } 32... Rg6 { [%clk 0:00:28] } 33. Rdf1 { [%clk 0:01:16] } 33... g3 { [%clk 0:00:25] } 34. Nf4 { [%clk 0:01:15] } 34... Nxf4 { [%clk 0:00:23] } 35. Rxf4 { [%clk 0:01:15] } 35... g2 { [%clk 0:00:22] } 36. Rf2 { [%clk 0:01:12] } 36... Rg5 { [%clk 0:00:19] } 37. e6 { [%clk 0:01:11] } 37... Re5 { [%clk 0:00:18] } 38. Rgxg2+ { [%clk 0:01:09] } 38... Bxg2 { [%clk 0:00:12] } 39. Rxg2+ { [%clk 0:01:09] } 39... Kf6 { [%clk 0:00:09] } 40. Rh2 { [%clk 0:01:08] } 40... Kxe6 { [%clk 0:00:08] } 41. Rh6+ { [%clk 0:01:07] } 41... Kd5 { [%clk 0:00:05] } 42. Rxb6 { [%clk 0:01:07] } 42... Kc4 { [%clk 0:00:05] } 43. b3+ { [%clk 0:01:05] } 43... Kc3 { [%clk 0:00:03] } 44. Rc6+ { [%clk 0:01:05] } 44... Kb4 { [%clk 0:00:02] } 45. Kb2 { [%clk 0:01:04] } 1-0";
        super::parse_pgn(src).unwrap();
    }
}
