import sys


class AllAllowedChartConstraints:
    def is_edge_allowed(self, sentence_index, start, end):
        return True


class BeginEndChartConstraints:
    def __init__(self, bconst_filename, econst_filename):
        self.bconst = []
        self.econst = []

        with open(bconst_filename, "r") as fbconst:
            with open(econst_filename, "r") as feconst:
                for lbconst, leconst in zip(fbconst, feconst):
                    bc = lbconst.strip().split()
                    bc_bool = [x[0] != "~" for x in bc]
                    self.bconst.append(bc_bool)

                    ec = leconst.strip().split()
                    ec_bool = [x[0] != "~" for x in ec]
                    self.econst.append(ec_bool)

                    # if len(self.bconst) <= 2:
                    #     print(self.bconst[len(self.bconst)-1])
                    #     print(self.econst[len(self.econst) - 1])
                    #     print([len(x) for x in self.bconst])
                    #     print([len(x) for x in self.econst])

        print(f"Read BECC: {len(self.bconst)}, {len(self.econst)}")

    def is_edge_allowed(self, sentence_index, start, end):
        # print(f"allowed({sentence_index}):")
        # print(f"start {start}/{len(self.bconst[sentence_index])}, end {end}/{len(self.econst[sentence_index])}")

        if end > len(self.econst[sentence_index]):
            # This can occasionally happen, because of mismatch between Stanford and NLTK tokenizer
            print(f"WARNING: Requested end position {end} for sentence_index {sentence_index}, but len is only {len(self.econst[sentence_index])}")
            return False

        end_allowed = True if end == len(self.econst[sentence_index]) else self.econst[sentence_index][end]

        return self.bconst[sentence_index][start] and end_allowed

