use cozy_chess::{Board, Color, File, GameStatus, Move, Piece, Rank, Square};
use std::io::{self, Write};

const LOGO: &str = r#"
████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███████╗██╗███████╗██╗  ██╗
╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝██║██╔════╝██║  ██║
   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝█████╗  ██║███████╗███████║
   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██╔══╝  ██║╚════██║██╔══██║
   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║     ██║███████║██║  ██║
   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
"#;

// Changed arguments to take Options, as that's what cozy-chess returns
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
        print!("{} | ", rank + 1); // Rank number

        for file in 0..8 {
            let square = Square::new(File::index(file), Rank::index(rank));
            let piece = board.piece_on(square);
            let color = board.color_on(square);

            // Pass the options directly
            print!("{}", get_unicode_piece(piece, color));
        }
        println!("| {}", rank + 1);
    }

    println!("  +-----------------");
    println!("    a b c d e f g h\n");
}

fn main() -> io::Result<()> {
    println!("{}", LOGO);

    let mut player_colour = String::new();
    let tensorfish_colour: &str;

    loop {
        print!("Choose colour (w/b): ");
        io::stdout().flush()?;

        player_colour.clear();
        io::stdin().read_line(&mut player_colour)?;

        match player_colour.trim() {
            "w" => {
                tensorfish_colour = "b";
                break;
            }
            "b" => {
                tensorfish_colour = "w";
                break;
            }
            _ => println!("Invalid choice, please type 'w' or 'b'."),
        }
    }

    println!("Tensorfish plays as: {}", tensorfish_colour);

    let mut board = Board::default();
    while board.status() == GameStatus::Ongoing {
        print_board(&board);
        
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
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            // Try to parse the UCI string (e.g., "e2e4" or "a7a8q")
            match input.parse::<Move>() {
                Ok(m) => {
                    // Check if this move is actually legal in the current position
                    if move_list.contains(&m) {
                        player_move_obj = m;
                        break; // Move is valid, exit input loop
                    } else {
                        println!("Invalid move: '{}' is not legal right now.", input);
                    }
                }
                Err(_) => {
                    println!("Invalid format: '{}'. Use UCI format like 'e2e4'.", input);
                }
            }
        }

        board.play(player_move_obj);
    }

    Ok(())
}
