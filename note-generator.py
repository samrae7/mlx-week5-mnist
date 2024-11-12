import itertools

class NoteGenerator():
    def __init__(self):
        self.genres = {
            'rock': ['C', 'E', 'G', 'F', 'A'],
            'jazz': ['D', 'G', 'A#', 'F', 'E', 'A', 'C#'],
            'classical': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'blues': ['C', 'E', 'G', 'A', 'B♭', 'D'],
        }

    def generate_notes_for_genre(self, genre):
        """Generate an infinite sequence of notes for the specified genre."""
        if genre not in self.genres:
            raise ValueError(f"Genre '{genre}' is not supported.")
        
        # infinitely cycles through the genre's notes
        genre_notes = itertools.cycle(self.genres[genre])
        
        while True:
            yield next(genre_notes)

    def generate_infinite_music(self, genre, length=8):
        """Generate a fixed length sequence of pseudo-music notes repeatedly."""
        note_generator = self.generate_notes_for_genre(genre)
        while True:
            # Generate a sequence of specified length, then repeat
            sequence = [next(note_generator) for _ in range(length)]
            yield sequence

if __name__ == "__main__":
    note_generator = NoteGenerator()
    rock_music = note_generator.generate_infinite_music('rock', length=8)
    
    # Print the first 3 sequences for demonstration
    for _ in range(3):
        print(next(rock_music))