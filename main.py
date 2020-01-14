import image_processing.image_proc as im_proc
import algorithmics.puzzle as puzzle


# Exciting! actual real main
# Need to think about shceme - maybe we need thread for greedy in terms of speed
# If it's slow, then greedy and execution are intertwined


def main():
    # Parse pieces
    pieces = im_proc.get_pieces(True)
    puzzle_obj = puzzle.Puzzle(pieces)

    # Build puzzle
    puzzle_obj.greedy()
    puzzle_obj.display()

    # Execute puzzle


if __name__ == "__main__":
    main()
