use cozy_chess::{BitBoard, Board, Color, File, GameStatus, Move, Piece, Rank, Square};
use ndarray::Array3;
use ort::{
    ep::CUDA,
    session::{Session, builder::GraphOptimizationLevel},
};
use std::io::{self, Write};

const LOGO: &str = r#"
████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███████╗██╗███████╗██╗  ██╗
╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝██║██╔════╝██║  ██║
   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝█████╗  ██║███████╗███████║
   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██╔══╝  ██║╚════██║██╔══██║
   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║     ██║███████║██║  ██║
   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
"#;

fn get_unicode_piece(piece: Option<Piece>, color: Option<Color>) -> String {
    match (color, piece) {
        (Some(Color::White), Some(p)) => match p {
            Piece::Pawn => "♟ ".to_string(),
            Piece::Knight => "♞ ".to_string(),
            Piece::Bishop => "♝ ".to_string(),
            Piece::Rook => "♜ ".to_string(),
            Piece::Queen => "♛ ".to_string(),
            Piece::King => "♚ ".to_string(),
        },
        (Some(Color::Black), Some(p)) => match p {
            Piece::Pawn => "♙ ".to_string(),
            Piece::Knight => "♘ ".to_string(),
            Piece::Bishop => "♗ ".to_string(),
            Piece::Rook => "♖ ".to_string(),
            Piece::Queen => "♕ ".to_string(),
            Piece::King => "♔ ".to_string(),
        },
        _ => ". ".to_string(), // Empty square
    }
}

fn print_board(board: &Board) {
    println!("\n    a b c d e f g h");
    println!("  +-----------------");

    for rank in (0..8).rev() {
        print!("{} | ", rank + 1);

        for file in 0..8 {
            let square = Square::new(File::index(file), Rank::index(rank));
            let piece = board.piece_on(square);
            let color = board.color_on(square);
            print!("{}", get_unicode_piece(piece, color));
        }

        println!("| {}", rank + 1);
    }

    println!("  +-----------------");
    println!("    a b c d e f g h\n");
}

fn get_player_move(board: &Board) -> Move {
    let mut move_list = Vec::new();
    board.generate_moves(|moves| {
        move_list.extend(moves);
        false
    });

    let move_strings: Vec<String> = move_list.iter().map(|m| m.to_string()).collect();
    println!("Valid moves: {}", move_strings.join(", "));

    let player_move_obj: Move;
    loop {
        print!("Your move (e.g. e2e4): ");
        io::stdout().flush().expect("Error flushing stdout");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Unable to read line");
        let input = input.trim();

        match input.parse::<Move>() {
            Ok(m) => {
                if move_list.contains(&m) {
                    player_move_obj = m;
                    break;
                } else {
                    println!("Invalid move: '{}' is not legal right now.", input);
                }
            }
            Err(_) => {
                println!("Invalid format: '{}'. Use UCI format like 'e2e4'.", input);
            }
        }
    }

    return player_move_obj;
}

fn get_tensorfish_move(board: &Board, model: &Session) -> Move {
    let depth = 4; // adjust | 4 is already very slow without optimisations
    let maximising = board.side_to_move() == Color::White;

    return minimax(board, depth, maximising, model)
        .1
        .expect("No legal moves");
}

fn file_to_idx(file: File) -> usize {
    match file {
        File::A => 0,
        File::B => 1,
        File::C => 2,
        File::D => 3,
        File::E => 4,
        File::F => 5,
        File::G => 6,
        File::H => 7,
    }
}

fn board_to_tensor(board: &Board) -> Array3<f32> {
    let mut tensor = Array3::<f32>::zeros((8, 8, 19));

    // Piece planes (0–11)
    for sq in Square::ALL {
        if let Some(piece) = board.piece_on(sq)
            && let Some(colour) = board.color_on(sq)
        {
            let plane = match (colour, piece) {
                (Color::White, Piece::Pawn) => 0,
                (Color::White, Piece::Knight) => 1,
                (Color::White, Piece::Bishop) => 2,
                (Color::White, Piece::Rook) => 3,
                (Color::White, Piece::Queen) => 4,
                (Color::White, Piece::King) => 5,
                (Color::Black, Piece::Pawn) => 6,
                (Color::Black, Piece::Knight) => 7,
                (Color::Black, Piece::Bishop) => 8,
                (Color::Black, Piece::Rook) => 9,
                (Color::Black, Piece::Queen) => 10,
                (Color::Black, Piece::King) => 11,
                _ => unreachable!(),
            };
            let (rank, file) = (sq.rank() as usize, sq.file() as usize);
            tensor[[rank, file, plane]] = 1.0;
        }
    }

    // Side to move (12)
    if board.side_to_move() == Color::White {
        tensor.slice_mut(ndarray::s![.., .., 12]).fill(1.0);
    }

    // Castling rights (13–16)
    let castling_rights_white = board.castle_rights(Color::White);
    let castling_rights_black = board.castle_rights(Color::Black);

    if castling_rights_white.short.unwrap() == File::H {
        tensor.slice_mut(ndarray::s![.., .., 13]).fill(1.0);
    }
    if castling_rights_white.long.unwrap() == File::A {
        tensor.slice_mut(ndarray::s![.., .., 14]).fill(1.0);
    }
    if castling_rights_black.short.unwrap() == File::H {
        tensor.slice_mut(ndarray::s![.., .., 15]).fill(1.0);
    }
    if castling_rights_black.long.unwrap() == File::A {
        tensor.slice_mut(ndarray::s![.., .., 16]).fill(1.0);
    }

    // En passant (17)
    if let Some(ep) = board.en_passant() {
        let rank: usize;
        let file = file_to_idx(ep);
        if board.side_to_move() == Color::White {
            rank = 6;
        } else {
            rank = 3;
        }

        tensor[[rank, file, 17]] = 1.0;
    }

    // Halfmove clock normalised (18)
    tensor
        .slice_mut(ndarray::s![.., .., 18])
        .fill(board.halfmove_clock() as f32 / 100.0);

    return tensor;
}

fn evaluate(board: &Board, model: &Session) -> f32 {
    let input = board_to_tensor(board);
    let input_view = input.view().into_dyn();
    let outputs = model.run(ort::inputs!["input" => input_view]).unwrap();
    let output = outputs[0].try_extract_tensor::<f32>().unwrap();
    return output.1[0];
}

fn minimax(board: &Board, depth: u32, maximising: bool, model: &Session) -> (i32, Option<Move>) {
    if board.status() == GameStatus::Drawn {
        return (0, None);
    } else if board.checkers() != BitBoard::EMPTY {
        if board.side_to_move() == Color::White {
            return (i32::MAX, None); // checkmate for white
        } else {
            return (i32::MIN, None); // checkmate for black
        }
    } else if depth == 0 {
        return (evaluate(&board.clone(), model), None);
    }

    let mut move_list = Vec::new();
    board.generate_moves(|moves| {
        move_list.extend(moves);
        false
    });

    let mut cloned_board = board.clone();
    let mut best_move = None;
    let mut best_value;

    if maximising {
        best_value = i32::MIN;
        for &m in move_list.iter() {
            cloned_board.play(m);
            let (value, _) = minimax(&cloned_board, depth - 1, false, model);
            if value > best_value {
                best_value = value;
                best_move = Some(m);
            }
        }
    } else {
        best_value = i32::MAX;
        for &m in move_list.iter() {
            cloned_board.play(m);
            let (value, _) = minimax(&cloned_board, depth - 1, true, model);
            if value < best_value {
                best_value = value;
                best_move = Some(m);
            }
        }
    }

    return (best_value, best_move);
}

fn load_onnx_model() -> ort::Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([CUDA::default().build()])?
        .commit_from_file("../models/tensorfish.onnx")?;

    Ok(session)
}

fn main() -> io::Result<()> {
    println!("{}", LOGO);

    let model: Session;
    match load_onnx_model() {
        Ok(session) => {
            model = session;
        }
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    }

    let player_side: Color;
    let mut player_input = String::new();

    loop {
        print!("Choose colour (w/b): ");
        io::stdout().flush()?;
        player_input.clear();
        io::stdin().read_line(&mut player_input)?;

        match player_input.trim().to_lowercase().as_str() {
            "w" => {
                player_side = Color::White;
                break;
            }
            "b" => {
                player_side = Color::Black;
                break;
            }
            _ => println!("Invalid choice, please type 'w' or 'b'."),
        }
    }

    println!("You are playing as: {:?}", player_side);
    println!("Tensorfish is playing as: {:?}", !player_side);

    let mut board = Board::default();
    while board.status() == GameStatus::Ongoing {
        print_board(&board);

        let side_to_move = board.side_to_move();
        println!("Current turn: {:?}", side_to_move);

        let mv: Move;
        if player_side == side_to_move {
            mv = get_player_move(&board);
        } else {
            mv = get_tensorfish_move(&board.clone(), &model);
        }

        board.play(mv);
    }

    Ok(())
}
