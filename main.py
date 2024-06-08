import socket
import time
import chess
import stockfish
import numpy as np
from enum import Enum
from vision_board import VisionBoard
from icecream import ic
from move_bot import URMover
import random

# urm for moving the robot
urm = None

# Game State
class GameState(Enum):
    SETUP_BOARD = "setup board"
    WAIT_MOVE = "wait for player move"
    READ_MOVE = "read player move"
    MAKE_MOVE = "make robot move"
    FINISH_GAME = "finish game"


GAME_STATE = GameState.SETUP_BOARD
PLAYER_COLOR = chess.WHITE
# True = Black, False = White

# Physical variables
grab_height = 0.015 # !!! figure this height out so that robot doesn't break pieces/board

# Vision Variables
# Phone camera for now
CAM_PATH = "http://192.168.56.11:4747/video?1920x1080"
ROBOT_IP = "192.168.1.105"

# Robot functions
def cb_move(move: str, board: chess.Board):
    # is en passante?
    if "3" in move or "6" in move:
        if board.ep_square and chess.square_name(board.ep_square) in move:
            for i in range(-1, 2, 2):
                if chess.square_name(board.ep_square + i * 8) in move:
                    ic("EN PASSANTE WOOOOOO")
                    _cb_remove(chess.square_name(board.ep_square + i * 8))
                    _cb_move(move[:2], move[2:4])
                    break
    # is castling??
    if "e1" in move or "e8" in move:
        ic("Castling")
        rank = move[1]
        if f"c{rank}" in move or f"g{rank}" in move:
            is_long = move[2] == "c"
            # move rook
            _cb_move(
                f"a{rank}" if is_long else f"h{rank}",
                f"d{rank}" if is_long else f"f{rank}",
            )
            # move king
            _cb_move(move[:2], move[2:4])
    # promotion ??
    if len(move) == 5:
        ic("Promotion")
        if board.piece_at(chess.parse_square(move[2:4])) is not None:
            _cb_remove(move[2:4])
            print(f"Took {move[2:4]} and...")
        _cb_remove(move[:2])
        print(f"removed pawn now promoting...")
        _ = input(f"Please put {move[4]} on {move[2:4]}...")
    # generic move
    elif len(move) == 4:
        ic("Generic move...")
        if board.piece_at(chess.parse_square(move[2:4])) is not None:
            _cb_remove(move[2:4])
            ic(f"... with capture of {board.piece_at(chess.parse_square(move[2:4])).symbol()} on {move[2:4]}")
        _cb_move(move[:2], move[2:4])


def _cb_move(sq1: str, sq2: str):
    if urm is None:
        print("URM UNSET!!!!!!!!!!!!")
        return
    urm.move_grip(sq1, sq2, grab_height, grab_height)
grave = "]4" # this is to the left of the board
def _cb_remove(sq: str):
    _cb_move(sq, grave)


# Main Loop
def main():

    global PLAYER_COLOR
    global GAME_STATE
    PLAYER_COLOR = (
        chess.WHITE
        if input(
            "Enter your color: \n[W/w/Y/y/White/white]->White, \n[Anything else]->Black\n"
        )
        in ["W", "w", "Y", "y", "White", "white"]
        else chess.BLACK
    )
    print("The player is " + "white." if PLAYER_COLOR else "black.")
    board = chess.Board()
    sf = stockfish.Stockfish()
    sf.set_fen_position(board.fen())

    # Setup vision
    bv = VisionBoard(CAM_PATH)
    global urm
    urm = URMover(ROBOT_IP)
    if GAME_STATE == GameState.SETUP_BOARD:
        bv.setup()

        # Read initial board maybe
        _ = input("Please arrange the board into starting position and press ENTER...")
        board = bv.read_board()
        if input("Do you wish to input starting FEN manually [Yy/Nn anything else] : ") in ['Y', 'y']:
            inp_fen = input("ENTER FEN : ")
            board.set_board_fen(inp_fen)
            bv.snap_tiles()
        print("start :\n", board)
        sf.set_fen_position(board.fen())
        # Set all possible castling rights
        board.castling_rights |= chess.BB_H1 | chess.BB_H8 | chess.BB_A1 | chess.BB_A8

    GAME_STATE = (
        GameState.READ_MOVE if PLAYER_COLOR == chess.WHITE else GameState.MAKE_MOVE
    )
    print(board.unicode())
    while True:
        if board.turn == PLAYER_COLOR:
            if GAME_STATE == GameState.READ_MOVE:
                move = None
                waiting = True
                while waiting:
                    # wait for input indicating move was made
                    _ = input("Press ENTER when you've made a legal move.")
                    move = bv.read_move(board)
                    ic(move)
                    if move is None:
                        print("Illegal move, try again...")
                    else:
                        waiting = False
                if move is not None:
                    board.push(move)
                    GAME_STATE = GameState.MAKE_MOVE
        elif board.turn != PLAYER_COLOR and GAME_STATE == GameState.MAKE_MOVE:
            print("SF moves...")
            move = random.choice(list(board.legal_moves))
            sf_move_uci = str(move)
            ic(sf_move_uci, "to be made by robot.")
            cb_move(sf_move_uci, board)
            board.push_uci(sf_move_uci)
            # sf.set_fen_position(board.fen())
            bv.snap_tiles()

            print(board)
            bv.snap()
            GAME_STATE = GameState.READ_MOVE
        print(board)
        # sf.set_fen_position(board.fen())
        if board.is_game_over():
            print(f"Game finished.")
            break


if __name__ == "__main__":
    main()
