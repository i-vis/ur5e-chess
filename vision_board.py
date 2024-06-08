from logging import warning
import cv2
import numpy as np
import math
import time
import bl_video_capture
from enum import Enum
from icecream import ic
import random
from functools import cmp_to_key
from ultralytics import YOLO
import chess

# constants
BOARD_SIZE = [8, 8] # board size in cells
P_BOARD_SIZE = [810, 810] # pixel board size
YOLO_MODEL = "./yolo/runs/classify/train6/weights/best.pt"

def _cmp_two_points(a,b):
    if abs(a[1]-b[1]) < P_BOARD_SIZE[0]//BOARD_SIZE[0]//2:
        return a[0] - b[0]
    return a[1]-b[1]

class Board_State(Enum):
    PICK_CORNER_TL = "picking corner tl"
    PICK_CORNER_BR = "picking corner br"
    CHOSE_CORNERS = "chose all corners"
    FINISHED = "finished"

class VisionTile:
    name: str
    c: list
    tl: list
    tr: list
    bl: list
    br: list
    prev_img = None
    cur_img = None
    _snaps = 0
    
    def __repr__(self):
        return f"(VT:{self.name}, {self.c})"

    def snap(self, w_img, save:bool = True, snap_path:str = "") -> list:
        p1 = np.float32([self.tl, self.tr, self.bl, self.br])
        p2 = np.float32([[0,0], [P_BOARD_SIZE[0]//(BOARD_SIZE[0]+1), 0], [0, P_BOARD_SIZE[1]//(BOARD_SIZE[1]+1)], [P_BOARD_SIZE[0]//(BOARD_SIZE[0]+1), P_BOARD_SIZE[1]//(BOARD_SIZE[1]+1)]])
        M = cv2.getPerspectiveTransform(p1, p2)
        res = cv2.warpPerspective(w_img, M, (P_BOARD_SIZE[0]//(BOARD_SIZE[0]+1),P_BOARD_SIZE[1]//(BOARD_SIZE[1]+1)))

        
        if save:
            self.cur_img = res
            if not self.cur_img is None:
                self.prev_img = self.cur_img.copy()

        # debug
        name = f"{self.name}" + (f"_{self._snaps}" if self._snaps > 0 else "") + ".png"
        if snap_path != "":
            cv2.imwrite(f"{snap_path}", self.cur_img)
        else:
            cv2.imwrite(f"./yolo/debug/{name}", self.cur_img)
        self._snaps += 1

        return res

    def check_diff(self, w_img, yolo: YOLO, board)->bool:
        res = None
        if self.prev_img is None or self.cur_img is None:
            return False
        if yolo is None:
            raise Exception("YOLO was not passed")
        tmp = self.snap(w_img, save = False)

        yolo_res, yolo_res2 = yolo(self.prev_img)[0], yolo(tmp)[0]
        prev = yolo_res.names[yolo_res.probs.top1]
        prev_b = board.piece_at(chess.parse_square(self.name))
        if prev_b is None:
            prev_b = "empty"
        else:
            prev_b = prev_b.symbol()
        cur = yolo_res2.names[yolo_res2.probs.top1]
        res = not prev_b == cur
        if cur.isupper() == prev_b.isupper() and cur != "empty" and prev_b != "empty": # Prevent K Q situation
            return False
        ic(res, prev, prev_b, cur, self.name)
        return res

class VisionBoard:
    cam_path : str
    chess_tiles : dict
    cam : bl_video_capture.VideoCapture
    board_state = Board_State.PICK_CORNER_TL
    _board_coords : list
    yolo : YOLO
    crop_coords : list

    def __init__(self, cam_path):
        self.cam_path = cam_path
        self.chess_tiles = {}
        self.yolo = YOLO(YOLO_MODEL)
        # ic(self.yolo)

    def snap(self):
        w_img = self.read_crop_board()
        for i in self.chess_tiles.values():
            i.snap(w_img)

    def read_crop(self) -> np.ndarray:
        if self.crop_coords is None:
            warning("Crop coords were not set..")
            raise Exception("Crop coords were not set")
        img = None
        _, img = self.cam.read()
        if img is None: raise Exception()
        return img[self.crop_coords[0][1] : self.crop_coords[3][1], self.crop_coords[0][0] : self.crop_coords[3][0]]

    def read_crop_board(self) -> np.ndarray: 
        if self._board_coords is None:
            warning("Board coords were not set..")
            raise Exception("Board coords were not set")
        img = self.read_crop()
        pos1 = self._board_coords
        pos2 = np.float32([[0, 0], [P_BOARD_SIZE[0], 0], [0, P_BOARD_SIZE[1]], P_BOARD_SIZE])
        mat = cv2.getPerspectiveTransform(pos1, pos2)
        w_img = cv2.warpPerspective(img, mat, P_BOARD_SIZE)
        return w_img

    def setup(self) -> bool:
        self.cam = bl_video_capture.VideoCapture(self.cam_path)
        _, img = self.cam.read()

        # Crop video feed REDUNDANT TR, BL
        tl = [0, 0]; br = [0, 0]; tr = [0, 0]; bl = [0, 0]
        ntl = [0, 0]; nbr = [0, 0]; ntr = [0, 0]; nbl = [0, 0]
        def _read_pos(event, x, y, flags, param):
            nonlocal tr, tl, br, bl
            if event == cv2.EVENT_LBUTTONDBLCLK:
                print(x, y)
                match self.board_state:
                    case Board_State.PICK_CORNER_TL:
                        tl = [x, y]
                        self.board_state = Board_State.PICK_CORNER_BR
                    case Board_State.PICK_CORNER_BR:
                        br = [x, y]
                        self.board_state = Board_State.CHOSE_CORNERS
                    case _:
                        print(f"Something went wrong : {self.board_state}")
        cv2.namedWindow('choose')
        cv2.setMouseCallback('choose', _read_pos)
        while self.board_state != Board_State.CHOSE_CORNERS:
            cv2.imshow('choose', img)
            k = cv2.waitKey(20) & 0xFF
        tr = [br[0], tl[1]]; bl = [tl[0], br[1]]
        self.crop_coords = [tl, tr, bl, br]
        ic(self.crop_coords)
        ntl = [0, 0]; ntr = [br[0]-tl[0], 0]; nbl = [0, br[1]-tl[1]]; nbr = [br[0]-tl[0], br[1]-tl[1]]
        ic(f"{tl}, {tr}, {bl}, {br}")
        ic(f"{ntl}, {ntr}, {nbl}, {nbr}")
        cv2.destroyWindow("choose")

        # Try to find suitable frame for setup
        success = False
        intersects = []
        while not success:
            img = self.read_crop()

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh_img = 255-thresh_img
            kernel = np.ones((5, 5), np.uint8)
            thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)
        

            contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contour = max(contours, key = cv2.contourArea)
            # ic(contour)
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            # cv2.drawContours(thresh_img, [approx], -1, (255,0,255), 2)
            approx = [e[0] for e in approx]
            ic(approx)
            approx = sorted(approx, key=cmp_to_key(_cmp_two_points))
            ic(approx)

            cv2.imshow("test_thresh", thresh_img)
            cv2.waitKey(0)

            pos1 = np.float32([approx[0], approx[1], approx[2], approx[3]])
            ic(pos1)
            self._board_coords = pos1
            w_img = self.read_crop_board()
            # w_thr_img = cv2.warpPerspective(thresh_img, mat, P_BOARD_SIZE) # ? needed
            points_img = np.zeros((P_BOARD_SIZE[0], P_BOARD_SIZE[1],3), np.uint8) + 255
            
            edges = cv2.Canny(w_img, 40, 170)
            edges = cv2.dilate(edges, kernel, iterations=1)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 300, None)
            cv2.imshow("EDGES", edges)

            # skip frames with no lines
            if lines is None: continue 

            # split lines into two categories, horizontal and vertical lines
            hor_lines = []; ver_lines = [];
            for line in lines:
                rho, theta = line[0]
                if abs(theta - 0) > abs(theta - math.pi/2):
                    # only accept 90+-1 degree lines
                    if abs(theta - math.pi/2) < math.pi/180:
                        ver_lines.append(line)
                else:
                    # only accept 0+-1 degree lines
                    if abs(theta) < math.pi/180:
                        hor_lines.append(line)
            def hough_inter(rho1, theta1, rho2, theta2):
                A = np.array([[math.cos(theta1), math.sin(theta1)],
                              [math.cos(theta2), math.sin(theta2)]])
                b = np.array([rho1, rho2])
                return np.linalg.lstsq(A, b)[0]

            intersects = []
            for i in hor_lines:
                i = i[0]
                x0 = math.cos(i[1]) * i[0]
                y0 = math.sin(i[1]) * i[0]
                pt1 = (int(x0 + 1000*(-math.sin(i[1]))), int(y0 + 1000*(math.cos(i[1]))))
                pt2 = (int(x0 - 1000*(-math.sin(i[1]))), int(y0 - 1000*(math.cos(i[1]))))
                cv2.line(w_img, pt1, pt2, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3, cv2.LINE_AA)
                for j in ver_lines:
                    j = j[0]
                    x0 = math.cos(j[1]) * j[0]
                    y0 = math.sin(j[1]) * j[0]
                    pt1 = (int(x0 + 1000*(-math.sin(j[1]))), int(y0 + 1000*(math.cos(j[1]))))
                    pt2 = (int(x0 - 1000*(-math.sin(j[1]))), int(y0 - 1000*(math.cos(j[1]))))
                    cv2.line(w_img, pt1, pt2, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3, cv2.LINE_AA)
                    r = hough_inter(i[0], i[1], j[0], j[1])
                    r = [int(r[0]), int(r[1])]
                    intersects.append(r)

            # some additional processing for intersects
            tmp = []
            for i in intersects:
                def dist(a, b):
                    return math.hypot(abs(a[0]-b[0]), abs(a[1]-b[1]))
                # check if already processed
                present_in_tmp = False
                for k in tmp:
                    if dist(i, k) < 20:
                        present_in_tmp = True
                        break
                if present_in_tmp: continue
                # count close points and get average
                close_points = []
                for j in intersects:
                    if dist(i, j) < 20:
                        close_points.append(j)
                if len(close_points) > 0:
                    newx = i[0]; newy = i[1]
                    for k in close_points:
                        newx += k[0]; newy += k[1]
                    newx //= len(close_points)+1; newy //= len(close_points)+1
                    i = [newx, newy]
                tmp.append(i)
            intersects = []
            # ic(min(nbr)//8//2)
            edge_distance = [P_BOARD_SIZE[0]//BOARD_SIZE[0]//3, P_BOARD_SIZE[1]//BOARD_SIZE[1]//3]
            # edge_distance = [40, 40]
            # ic(edge_distance)
            # edge_distance = [nbr[0]//BOARD_SIZE[0]//3, nbr[1]//BOARD_SIZE[1]//3]
            # ic(edge_distance)
            for i in tmp:
                if i[0] < edge_distance[0] or i[0] > P_BOARD_SIZE[0] - edge_distance[0] or i[1] < edge_distance[1] or i[1] > P_BOARD_SIZE[1] - edge_distance[1]:
                    continue
                intersects.append(i)

            for i in intersects:
                points_img = cv2.circle(points_img, i, 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), random.randint(3, 7))
            
            cv2.imshow("warp_img", w_img)
            # cv2.imshow("warp_thresh", w_thr_img)
            cv2.imshow("points", points_img)
            
            cv2.imshow("image", img)
            # cv2.imshow("src", src)
            cv2.waitKey(0)
            
            # check how many intersects and if we got correct result or not
            ic(len(intersects))
            if (len(intersects) == ((BOARD_SIZE[0]+1) * (BOARD_SIZE[1]+1))):
                _ = input("SUCCESS")
                success = True
        
        # sort intersects
        # intersects = sorted(intersects, key=lambda e: (e[1], e[0]))
        intersects = sorted(intersects, key=cmp_to_key(_cmp_two_points))

        ic(intersects)

        for i in range(0, BOARD_SIZE[0]):
            for j in range(0, BOARD_SIZE[1]):
                new = VisionTile()
                new.name = chr(ord('a') + i)+str(1+j)
                new.tl = intersects[(i)*(BOARD_SIZE[0]+1)   +j]
                new.tr = intersects[(i+1)*(BOARD_SIZE[0]+1) +j]
                new.bl = intersects[(i)*(BOARD_SIZE[0]+1)   +(j+1)]
                new.br = intersects[(i+1)*(BOARD_SIZE[0]+1) +(j+1)]
                new.c = [(new.tl[0] + new.tr[0] + new.bl[0] + new.br[0]) // 4, (new.tl[1] + new.tr[1] + new.bl[1] + new.br[1]) // 4]
                self.chess_tiles[new.name] = new
                w_img = self.read_crop_board()
                # pos1 = np.float32([tl, tr, bl, br])
                # pos2 = np.float32([ntl, ntr, nbl, nbr])
                # mat = cv2.getPerspectiveTransform(pos1, pos2)
                # img = cv2.warpPerspective(img, mat, nbr)
                # pos1 = self._board_coords
                # pos2 = np.float32([[0, 0], [P_BOARD_SIZE[0], 0], [0, P_BOARD_SIZE[1]], P_BOARD_SIZE])
                # mat = cv2.getPerspectiveTransform(pos1, pos2)
                # w_img = cv2.warpPerspective(img, mat, P_BOARD_SIZE)
                new.snap(w_img)
                ic(new)
        
        # DEBUG
        # Take more pictures for making dataset DEBUG
        make_dataset = False
        # make_dataset = True
        if make_dataset:
            board = None
            while True:
                fen = input("Type FEN for making dataset : ")
                try:
                    board = chess.Board(fen)
                    break
                except:
                    print("Invalid FEN, try again.")
                    continue
            i_shift = 0; j_shift = 0
            inp = ""
            while inp != "DONE":
                w_img = self.read_crop_board()
                cv2.imshow("SNAPIMG", w_img)
                cv2.waitKey()

                print("Pressing enter will make a new snap for dataset. Typing anything will discard photo as bad. When you are done type \"DONE\"...")
                inp = input() 
                if len(inp) > 0:
                    continue
                
                file_dir = "./yolo/latest_dataset"
                for i in self.chess_tiles:
                    i = self.chess_tiles[i]
                    sq = chess.parse_square(i.name)
                    orig_sq = sq // 8 * 8 + (sq - j_shift) % 8      # shift by j_shift
                    orig_sq = (orig_sq - 8 * i_shift) % 64          # shift by i_shift

                    file_name = ""
                    piece = ""
                    if board.piece_at(orig_sq):
                        piece = board.piece_at(orig_sq).symbol()
                    file_name = i.name + f"_{i_shift}_{j_shift}" + ".png"
                    if piece == "": piece = "black" if (ord(i.name[0]) + ord(i.name[1])) % 2 == 1 else "white"
                    ic(f"{file_dir}/{piece}/{file_name}")
                    i.snap(w_img, snap_path=f"{file_dir}/{piece}/{file_name}")
                i_shift += 1
                j_shift += i_shift // 8
                i_shift %= 8

        # self.read_board()

        return True

    def read_board(self) -> chess.Board:
        result = chess.Board.empty()
        for square_name in self.chess_tiles:
            tile = self.chess_tiles[square_name]
            w_img = self.read_crop_board()

            tile_img = tile.snap(w_img)
            yolo_res = self.yolo(tile_img)[0]
            piece_class = yolo_res.names[yolo_res.probs.top1]
            ic(piece_class)
            ic([yolo_res.names[i] for i in yolo_res.probs.top5])
            if piece_class in ["black", "white", "empty"]: continue

            result.set_piece_at(chess.parse_square(square_name), chess.Piece.from_symbol(piece_class)) 
        ic(result)
        ic(result.unicode())
        return result
    
    def snap_tiles(self):
        w_img = self.read_crop_board()
        cv2.imshow("confirm", w_img)
        cv2.waitKey()
        inp = ""
        while inp == "":
            inp = input("Does the image look good, i.e. no external objects? type anything to continue")
            w_img = self.read_crop_board()
            cv2.imshow("confirm", w_img)
            cv2.waitKey()
        for i in self.chess_tiles.values():
            i.snap(w_img, save=False)

    def read_move(self, board: chess.Board) -> chess.Move | None:
        result = ""

        changes = []
        w_img = self.read_crop_board()
        cv2.imshow("confirm", w_img)
        cv2.waitKey()
        for i in self.chess_tiles.values():
            if i.check_diff(w_img, self.yolo, board):
                changes.append(i)
        if not changes:
            return None
        ic(changes)
        changes = [ct.name for ct in changes]

        # 2 squares move
        if len(changes) == 2:
            uci = "".join(changes[::])
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves: 
                for c in changes:
                    self.chess_tiles[c].snap(w_img, save = False)
                return move

            # promotion
            piece_promoting = board.piece_at(chess.parse_square(changes[0]))
            if changes[1][1] in ["1", "8"] and piece_promoting and piece_promoting.symbol().lower() == "p":
                pclass = self.yolo(self.chess_tiles[changes[1]].snap(w_img, save=False))[0]
                pclass = pclass.names[pclass.probs.top1]
                pclass = pclass.lower() # bcs UCI notation is strange
                if len(pclass) > 1: return None
                move = chess.Move.from_uci(uci + f"{pclass}")
                if move in board.legal_moves:
                    for c in changes:
                        self.chess_tiles[c].snap(w_img, save = False)
                    return move

            uci = "".join(changes[::-1])
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves: 
                for c in changes:
                    self.chess_tiles[c].snap(w_img, save = False)
                return move

            # promotion
            piece_promoting = board.piece_at(chess.parse_square(changes[1]))
            if changes[0][1] in ["1", "8"] and piece_promoting and piece_promoting.symbol().lower() == "p":
                pclass = self.yolo(self.chess_tiles[changes[0]].snap(w_img, save=False))[0]
                pclass = pclass.names[pclass.probs.top1]
                pclass = pclass.lower() # bcs UCI notation is strange
                if len(pclass) > 1: return None
                move = chess.Move.from_uci(uci + f"{pclass}")
                if move in board.legal_moves:
                    for c in changes:
                        self.chess_tiles[c].snap(w_img, save = False)
                    return move
        
        # 3 squares move (should be enpassante)
        if len(changes) == 3:
            if board.ep_square:
                ep_name = chess.square_name(board.ep_square)
                if ep_name in changes:
                    for i in changes:
                        if ep_name == i: continue
                        uci = i + ep_name
                        move = chess.Move.from_uci(uci)
                        if move in board.legal_moves:
                            for c in changes:
                                self.chess_tiles[c].snap(w_img, save = False)
                            return move
                        
        # 4 squares move (castling only pretty much)
        if len(changes) == 4:
            # if king doesnt participate not castling
            king_pos = ""
            target_pos = ""
            for i in changes:
                if i[0] == "e":
                    king_pos = i
                if i[0] == "c":
                    target_pos = i
                if i[0] == "g":
                    target_pos = i
            if king_pos == "" or target_pos == "":
                return None
            uci = king_pos+target_pos
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return move


        # else move not detected
        return None
