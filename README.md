# scale invariant feature transform on gpu (pytorch extension)
original repo: https://github.com/Celebrandil/CudaSift<br/>
My result does not match the original perfectly<br/>

# ONLY 5ms(RTX 2080ti) to run. ENJOY !!!
python setup.py install

I added some constraints below.<br/>
It's at the bottom of the .cu file.<br/>

<pre>
<code>
int cnt = 0;
int num_matched_pts = 0;
float ctk1 = k;
float ctk2 = w-k-1;
float ctk3 = k;
float ckt4 = h-k-1;
float ctdstw = w * 0.15;
float ctdsth = h * 0.15;
for (int j=0;j<numPts;j++) { 
    if (sift1[j].match_error<5) {
        float x1 = sift1[j].xpos;
        float y1 = sift1[j].ypos;
        float x2 = sift1[j].match_xpos;
        float y2 = sift1[j].match_ypos;

        if (x1 < ctk1 || x1 > ctk2 || x2 < ctk1 || x2 > ctk2 || y1 < ctk3 || y1 > ckt4 || y2 < ctk3 || y2 > ckt4) {
            continue;
        }
        if (abs(x2 - x1) > ctdstw || abs(y2 - y1) > ctdsth) {
            continue;
        }

        matched_pts[cnt+0] = x1;
        matched_pts[cnt+1] = y1;
        matched_pts[cnt+2] = x2;
        matched_pts[cnt+3] = y2;
        cnt = cnt + 4;
        num_matched_pts = num_matched_pts + 1;
    }
}
</code>
</pre>

# Results Example
![left](https://github.com/sjg918/sift/blob/main/left.png?raw=true)
![right](https://github.com/sjg918/sift/blob/main/right.png?raw=true)
![result](https://github.com/sjg918/sift/blob/main/hi.png?raw=true)
