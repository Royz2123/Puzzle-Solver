import image_processing.image_proc as im_proc
import image_processing.util as util
import image_processing.camera as camera

import algorithmics.puzzle as puzzle

import mechanics.mechanics_api as mechanics_api

# Exciting! actual real main
# Need to think about shceme - maybe we need thread for greedy in terms of speed
# If it's slow, then greedy and execution are intertwined


def get_images(test=True):
    if test:
        above, below = util.get_test_images()
    else:
        above = camera.take_picture(still_camera=True)
        mechanics_api.send_command("U", 0)
        below = camera.take_picture(still_camera=True)
        mechanics_api.send_command("U", 1)

    return above, below


def main():
    # Parse pieces
    above, below = get_images(test=True)
    pieces = im_proc.get_pieces(above, below)
    puzzle_obj = puzzle.Puzzle(pieces)

    # Build puzzle
    puzzle_obj.greedy()
    puzzle_obj.display()

    # Execute puzzle


if __name__ == "__main__":
    main()
