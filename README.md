# UR5e Chess
Make UR5e play chess using image classification and stockfish.
This project is about making a cobot play chess using camera, and feeding board status to chess engine.

The project makes use of:
- [yolov8](https://docs.ultralytics.com/) for image classification
- [stockfish](https://pypi.org/project/stockfish/) for communication to [stockfish engine](https://en.wikipedia.org/wiki/Stockfish_(chess)), chess engine, that is one of the strongest chess engines in the world.
- [chess](https://pypi.org/project/chess/) python library for representing the board and as game engine for validating moves and checking more complex rules of chess.

## Installation
I suggest you make a [virtual environment](https://docs.python.org/3/library/venv.html). After you are set and done run `pip install -r requirements.txt`. You will also need to set up DroidCAM on your phone and set everything up physically. Change IPs in the code to whatever IPs you end up using, fiddle around with the robot to set up the correct translation of chess tiles to physical coordinates. After everything is set up you can run `python main.py`.

## Logic Flow
The camera takes a picture of the board from above to get the state of the board at certain times. An algorithm is applied to the board to determine the corners of the board and of individual chess squares. These chess squares' images are passed on to pretrained yoloV8 model for classification. This process is used for player move detection, initial board setup, and determining what piece pawn promote to. The program alternates states between reading player move, and making its' own move and sending commands to robot, until it determines that the game is over.

## Camera logic
After connecting to camera, user has an option to crop the image. We assume top-down view that won't change much over the course of the game. After that we run edge detection algorithm to get the corners of the chess board and warp perspective so that the image is only containing our chess boars. Then we get chess squares corners using Hough transform lines on the image that is pre-processed with adaptive Gaussian thresholding and getting interceptions between horizontal and vertical lines. We filter these interceptions, merging points that are very close into 1 point and removing points close to the edge. If we get 81 points we call that success and record the coordinates of these intercepts as cornwrs of individual tiles.
 
## Robot communication
 Communication with the robot is done via TCP commands on port 30002, via Ethernet link. There is an Ethernet port inside the robot's control box below the table with the robot. Commands are in special "urscript" that has [its' own manual](https://www.universal-robots.com/download/manuals-e-seriesur20ur30/script/script-manual-e-series-sw-511/).
 One of the things you should be aware of if you are going to be sending urscript over TCP is that every command is independent from the execution of others, so you need to send multiple commands at once.
 
## YoloV8
 It is a part of [ultralytics](https://pypi.org/project/ultralytics/) python package. I trained the model on my specific chess set and in home conditions and achieved pretty good results after trial-and-error for a week. The most challenging part for this was getting good photos of each tile in large quantities. I went with almost entirely manual approach for this, where the chessboard photos were automatically sliced, but I had to label the contents of the photo on my own. After getting solid dataset for training it was smooth sailing.

## Stockfish and "chess"
Honestly the easiest parts of this project. For stockfish you only need to install stockfish engine on your computer ( I downloaded stockfish package from arch repository) and learn how to use the library. For chess the only thing you need to do is learn how to utilise the classes and methods in the chess library. Of course it helps if you are familiar with how chess boards is represented and move notations. Here are some things that should help [chess rules](https://en.wikipedia.org/wiki/Rules_of_chess), [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface), [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).
