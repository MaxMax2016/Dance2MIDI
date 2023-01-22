# Dance2MIDI: Dance-driven multi-instruments music generation

### [Project Page](https://dance2midi.github.io/) | [Data](https://drive.google.com/drive/folders/1ZBeUciZWEZbLTwDz8keCRQI0Kw0BVgid?usp=share_link)

PyTorch implementation of Dance2MIDI

DANCE2MIDI: DANCE-DRIVEN MULTI-INSTRUMENTS MUSIC GENERATION

Bo Han,Yi Ren

Zhejiang University, Sea AI Lab

## setup

Set up a conda environment

```
'''
python 3.6
pip install -r requirement.txt
'''
```

## Download D2MIDI Dataset



Our D2MIDI Dataset has a total of 6000 pairs of data, in which the dance type includes classical dance, hip-hop, ballet, modern dance, and house dance. The music in each data pair does not repeat each other. In the D2MIDI dataset, the duration in each data pair is 30 seconds, which is guaranteed to generate music with a rhythmic structure. The music in the pair contains up to 12 tracks with 12 instrument types, including Acoustic Grand Piano, Celesta, Drawbar Organ, Acoustic Guitar (nylon), Acoustic Bass, Violin, String Ensemble 1, SynthBrass 1, Soprano Sax, Piccolo, Lead 1 (square), and Pad 1 (new age).



|        | Dance Video                                                  | Audio                                                        | MIDI                                                         |
| :----- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| D2MIDI | Five types of dance: classical dance,hip-hop, ballet, modern dance, and house dance | Each of them is non-repeating, and there are 6000 non-repeating pieces of music  audio in total | The MIDI music in the pair contains up to 12 tracks with 12 instrument types, including Acoustic Grand Piano, Celesta, Drawbar Organ, Acoustic Guitar (nylon), Acoustic Bass, Violin, String Ensemble 1, SynthBrass 1, Soprano Sax, Piccolo, Lead 1 (square), and Pad 1 (new age). |
| AIST   | Ten types of dance: Middle  Hip-hop, LA-style Hip-hop, House, Krump, Street Jazz, Ballet Jazz, Break, Pop, Lock, and Waack | There are a large number of repetitive pieces of music audio, and there are 60 non-repeating pieces of music  audio in total | No available                                                 |



For the released v1 version of D2MIDI, it contains 2692 pairs of data

# Train

```
python train.py
```

# Test

```
python test.py
```

