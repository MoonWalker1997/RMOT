# RMOT (not finished yet ⚠)

## Fulfil requirements

Run the following code.

```bash
$ pip install -r requirements.txt
```

## Data preparation

1. MOT17 official data.

   Download the MOT17 challenge official date (~5Gb) and extract the data under the directory`data/MOT17`.

   To make it clear, we may have:

   ```
   RMOT
   |——————data
          |——————MOT17
                 |——————MOT17
                        |——————test
                        |——————train
   ```

2. The tracking results (*.txt) you want to use as the base tracker to improve。

   Put the txt file under the directory `data/trackers`, if it is correct, you may see 4 built-in tracking results.

   ```
   RMOT
   |——————data
          |——————trackers
                 |—————MOT17-04-DPM-1.txt
                 |—————MOT17-04-DPM-2.txt
                 |—————MOT17-04-DPM-3.txt
                 |—————MOT17-04-DPM-4.txt
   ```

3. Download `data.zip` from https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md. Extract it and put in under the directory of `TrackEval-master`. It should be like.

   ```
   RMOT
   |——————TrackEval-master
          |——————data
                 |—————gt
                 |—————trackers
   ```

   

## Tracking improvement/evaluations

Go to the root directory `RMOT`, run the following code.

```bash
$ python3 hard_coded_samedet_offline_2.py -v MOT17-04-DPM -vl 1050 -f 30 -e MOT17-04-DPM-4 -e MOT17-04-DPM-4 -s res -it 0.7 -clt 0.3 -its 0.8 -at 0.6 -bt 0
```

If you just want to try a demo, just run the following code. It will use default arguments (the arguments shown above).

```bash
$ python3 hard_coded_samedet_offline_2.py
```

It you want to use your own arguments, please check the following description.

```bash
-v   | the name of the MOT17 challenge video you want to run
-vl  | the length of the above video, how many frames
-f   | the fps of the above video
-e   | the name of the external tracker result file
-s   | the name of the video generated and improved tracking results
-it  | the iou similarity threshold, from 0 to 1
-clt | the lower bound to say whether a box is occluded, from 0 to 1
-its | the stricter iou similarity threshold, from 0 to 1
-at  | the appearance similarity threshold, from 0 to 1
-bt  | the box score threshold, from 0 to 1
```

While running, you will see the progress of each frame and the fps of running.

After running, you will get a video and a txt file generated with the name under the direction specified in your arguments, then an evaluation will be printed on the screen.

If you want to check the details about the evaluation, please go to the following directory.

```bash
$ cd RMOT/TrackEval-master/data/trackers/mot_challenge/MOT17-train/hAIMOT
```

# Reference

1. TrackEval (https://github.com/JonathonLuiten/TrackEval/tree/master)
2. ONA (https://github.com/opennars/OpenNARS-for-Applications/tree/master)
3. MOT17 (https://motchallenge.net/data/MOT17/)
