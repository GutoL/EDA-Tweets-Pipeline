import numpy as np
import pandas as pd
import  math 
class PeakDetector():
    

    def lehmann_iteration(self, vec, i):
        N = len(vec)
        L = 30

        window_start_index = max(0, i - L)
        window_end_index = min(i + L, N)
        window = vec[window_start_index:window_end_index]
        current_activity = vec[i]
        median_activity = np.median(window)
        min_activity = 10
        outlier_fraction = (current_activity - median_activity) / max(median_activity, min_activity)
        threshold = 10

        return outlier_fraction > threshold

    def prune_peaks_lehmann(self, peaks):
        min_gap = 7
        if len(peaks) < 2:
            return peaks
        else:
            last_peak_index = peaks[0]
            pruned_peaks = [last_peak_index]
            for i in range(1, len(peaks)):
                if (peaks[i] - last_peak_index) > min_gap:
                    pruned_peaks.append(peaks[i])
                    last_peak_index = peaks[i]
            return pruned_peaks

    def find_peaks_lehmann(self, vec):
        peaks = []
        for i in range(len(vec)):
            if self.lehmann_iteration(vec, i):
                peaks.append(i+1)
        peaks = self.prune_peaks_lehmann(peaks)
        
        return peaks
    
    def find_peaks_palshikar(self, t, k, h, s):
        # Get significant function values for each xi in t
        n = len(t)
        a = [s(k, i, t[i], t) for i in range(n - 1)]
    
        # Get mean and standard deviation of positive values of a
        a_pos = [x for x in a if x > 0]
        m = sum(a_pos) / len(a_pos) if len(a_pos) > 0 else 0
        s = math.sqrt(sum((x - m) ** 2 for x in a_pos) / len(a_pos)) if len(a_pos) > 0 else 0
    
        hour = []
        tweets = []
        for i in range(n - 1):
            if a[i] > 0 and a[i] - m > h * s:
                hour.append(i)
                tweets.append(t[i])
    
        # Remove adjacent peaks
        unfiltered_output = pd.DataFrame({"hour": hour, "tweets": tweets})
        adjacency_group = pd.DataFrame({"hour": [], "tweets": []})
        filtered_output = pd.DataFrame({"hour": [], "tweets": []})
        for i in range(n - 2):
            # adjacency_group = adjacency_group.append(unfiltered_output.iloc[i:i+1, :])
            adjacency_group = pd.concat([adjacency_group, unfiltered_output.iloc[i:i+1, :]])

            if hour[i + 1] - hour[i] > k:
                # Get max row in adjacency group and add to output
                max_num_tweets_in_group = adjacency_group["tweets"].max()
                max_rows = adjacency_group[adjacency_group["tweets"] == max_num_tweets_in_group]
                
                # filtered_output = filtered_output.append(max_rows.iloc[0:1, :])
                filtered_output = pd.concat([filtered_output, max_rows.iloc[0:1, :]])
            
                # Reset adjacency group
                adjacency_group = pd.DataFrame({"hour": [], "tweets": []})
    
        return filtered_output.reset_index(drop=True)


    def palshikar_S1(self, k, i, xi, t):
        N = len(t)
        first_left_neighbour_index = max(0, i - k)
        last_left_neighbour_index = max(0, i - 1)
        first_right_neighbour_index = min(i + 1, N)
        last_right_neighbour_index = min(i + k, N)
    
        left_neighbours = t[first_left_neighbour_index:last_left_neighbour_index]
        right_neighbours = t[first_right_neighbour_index:last_right_neighbour_index]
    
        if len(left_neighbours) == 0:
            left_diff_max = 0
        else:
            left_diff_max = max(xi - left_neighbours)
    
        if len(right_neighbours) == 0:
            right_diff_max = 0
        else:
            right_diff_max = max(xi - right_neighbours)
    
        return (left_diff_max + right_diff_max) / 2

    def find_peaks_palshikar_s1(self, vec):
        output_df = self.find_peaks_palshikar(vec, k=12, h=3, s=self.palshikar_S1)
        print("Palshikar S1 output")
        print(output_df)
        return output_df["hour"]

