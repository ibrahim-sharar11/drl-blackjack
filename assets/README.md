Assets for Blackjack Viewer

Drop optional artwork and sounds here to enhance visuals and audio.

Recommended packs (public domain / CC0 from Kenney.nl):

- Playing Cards: https://kenney.nl/assets/playing-cards-pack
- Casino Pack (chips/table): https://kenney.nl/assets/casino-pack
- UI Sounds (optional): https://kenney.nl/assets/ui-audio

Folder structure (create if missing):

- assets/
  - cards/
    - cardSpadesA.png
    - cardSpades2.png
    - ...
    - cardHeartsK.png
    - cardBack_blue2.png (back)
  - chips/
    - chipRed.png, chipBlue.png, ... (any PNGs work)
  - table/
    - table.png (felt texture)
  - sounds/
    - deal.wav, flip.wav, win.wav, lose.wav, draw.wav (any of .wav/.ogg/.mp3)

Notes

- The viewer auto-detects assets; missing ones fall back to procedural drawings.
- Card name patterns supported:
  - Kenney: card{Suit}{Rank}.png (e.g., cardSpadesA.png, cardHearts10.png)
  - Generic: {rank}_of_{suit}.png (e.g., 10_of_spades.png, A_of_hearts.png)
  - Short: {rank}{SuitInitial}.png (e.g., AH.png, 10S.png)
- Sounds are optional; if present, they’ll play on deal/flip/win/lose/draw.

