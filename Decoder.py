import os
import numpy as np
import scipy
import copy
from PIL import Image
import pickle
import time
import matplotlib.pyplot as plt
import cv2

width = 352
height = 288
input_filepath = 'foreman_cif-1.yuv'
output_folder = 'Output/Exercise4/'

class Decoder:
    def __init__(self, frame_height, frame_width, block_size, QP, nRefFrames = 1,VBSEnable=False, lambda_map=None, FMEenable=False,
                 FastME=False):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.block_size = block_size
        self.QP = QP

        self.nRefFrames = nRefFrames
        initial_reference_frame = np.full((frame_height, frame_width), 128, dtype=np.uint8)
        self.reference_frames = [initial_reference_frame] * nRefFrames
        self.VBSEnable=VBSEnable
        self.lambda_map = lambda_map or {22: 0.1, 27: 0.05, 32: 0.01}
        self.possible_block_sizes = [4, 8, 16] if self.VBSEnable else [self.block_size]
        self.FMEenable = FMEenable
        self.FastME = FastME

        self.Q = np.zeros((block_size, block_size), dtype=np.uint16)
        for x in range(block_size):
            for y in range(block_size):
                if x + y < block_size - 1:
                    self.Q[x][y] = 2 ** QP
                elif x + y == block_size - 1:
                    self.Q[x][y] = 2 ** (QP + 1)
                else:
                    self.Q[x][y] = 2 ** (QP + 2)

    def reverse_entropy_coefficients(self, sequences):
        values = self.exponential_golomb_decoding(sequences)
        # print("values", values)
        RLE_decoded = self.RLE_sequence_decoding(values)
        # print("RLE_decoded", RLE_decoded)
        # reordering
        quant_trans_coefficients = []

        block_size = self.block_size
        matrix_height, matrix_width = block_size, block_size
        for RLE_decoded_sequence in RLE_decoded:
            i = 0
            quant_trans_coefficient = np.zeros(
                (matrix_height, matrix_width), dtype=np.int16)

            for line in range(1, (matrix_height + matrix_width)):
                start_col = max(0, line - matrix_height)
                count = min(line, (matrix_width - start_col), matrix_height)
                for j in range(0, count):
                    # print(start_col + j, min(matrix_height, line) - j - 1)
                    quant_trans_coefficient[start_col +
                                            j][min(matrix_height, line) - j - 1] = RLE_decoded_sequence[i]
                    i += 1

            quant_trans_coefficients.append(quant_trans_coefficient)

        return quant_trans_coefficients

    def rescaling(self, quant_trans_coefficients):
        # Inverse quantization operation
        transformed_coefficients = []
        for QTC in quant_trans_coefficients:
            TC = QTC * self.Q
            transformed_coefficients.append(TC)
        return transformed_coefficients

    def inverse_transform(self, transformed_coefficients):
        # Inverse 2D DCT
        residual_blocks = []
        for coefficient in transformed_coefficients:
            residual = scipy.fftpack.idct(scipy.fftpack.idct(
                coefficient, axis=0, norm='ortho'), axis=1, norm='ortho')
            residual_blocks.append(residual)
        return residual_blocks

    def reverse_entropy_predictions(self, sequences):
        # convert the sequences to array
        decoded_predictions = self.exponential_golomb_decoding(sequences)
        frame_marker, differential = decoded_predictions[0], decoded_predictions[1:]
        return frame_marker, differential

    def differential_decoding(self, frame_marker, differential):
        differential_decoded = []
        if frame_marker == 1:
            prev_mode = 0
            for diff_mode in differential:
                if diff_mode == 0:
                    differential_decoded.append(prev_mode)
                else:
                    differential_decoded.append(prev_mode + diff_mode)
                    prev_mode = prev_mode + diff_mode
        else:
            prev_mv = [0, 0, 0]
            for i in range(0, len(differential), 3):
                mv_y_diff, mv_x_diff, ref_index_diff = differential[i:i + 3]
                prev_mv_y, prev_mv_x, prev_ref_index = prev_mv

                differential_decoded.append(
                    [prev_mv_y + mv_y_diff, prev_mv_x + mv_x_diff, prev_ref_index + ref_index_diff])
                prev_mv = [prev_mv_y + mv_y_diff, prev_mv_x + mv_x_diff, prev_ref_index + ref_index_diff]
        return differential_decoded

    def RLE_sequence_decoding(self, sequences):
        # for all coeffcients
        RLE_decoded = []
        total = self.block_size * self.block_size

        curr = []
        i = 0
        while i < len(sequences):
            # non-zero elements
            if sequences[i] < 0:
                count = sequences[i] * -1
                i += 1

                for _ in range(count):
                    curr.append(sequences[i])
                    i += 1
            # zero elements
            elif sequences[i] > 0:
                count = sequences[i]
                i += 1
                for _ in range(count):
                    curr.append(0)
            # No further non-zero elements
            else:
                i += 1
                rest = total - len(curr)
                for _ in range(rest):
                    curr.append(0)
                RLE_decoded.append(curr)
                curr = []
                continue

            # handle case no 0
            if len(curr) >= total:
                RLE_decoded.append(curr)
                curr = []

        return RLE_decoded

    def exponential_golomb_decoding(self, sequences):
        values = []
        i = 0
        while i < len(sequences):
            zero_count = 0
            # count the number of 0
            while i < len(sequences) and sequences[i] == '0':
                zero_count += 1
                i += 1

            # read further binary string
            binary_count = zero_count + 1
            binary = ""
            while i < len(sequences) and binary_count > 0:
                binary += sequences[i]
                binary_count -= 1
                i += 1

            # convert to decimal
            decimal = int(binary, 2) - 1

            # convert back to number
            if decimal % 2 == 0:
                value = decimal // -2
            else:
                value = (decimal + 1) // 2
            values.append(value)
        return values

    def update_reference_frames(self, new_frame):
        self.reference_frames.append(copy.deepcopy(new_frame))
        if len(self.reference_frames) > self.nRefFrames:
            self.reference_frames.pop(0)

    def decode(self, output_folder):

        mdiff_file = output_folder + 'MDiff.txt'
        qtc_file = output_folder + 'QTC.txt'
        # output_file = output_folder + 'reconstructed_y_only_decoder'+str(self.nRefFrames)+'.y'
        output_file = output_folder + 'reconstructed_y_only_decoder.yuv'

        f_qtc = open(qtc_file, 'rb')
        f_mdiff = open(mdiff_file, 'rb')
        f_output = open(output_file, 'wb')

        frame_height, frame_width = self.frame_height, self.frame_width
        block_size = self.block_size

        decoded_frame = np.full((frame_height, frame_width), 128, dtype=np.uint8)

        frame_index = 0
        while True:
            try:
                # Read sequences from two files
                qtc_sequences = str(np.load(f_qtc, allow_pickle=True))
                mdiff_sequences = str(np.load(f_mdiff, allow_pickle=True))

                # Reverse Entropy first
                quant_trans_coefficients = self.reverse_entropy_coefficients(qtc_sequences)
                frame_marker, differential = self.reverse_entropy_predictions(mdiff_sequences)

                # Rescaling (reverse operation of quantization)
                transformed_coefficients = self.rescaling(quant_trans_coefficients)

                # Inverse DCT to get the residual blocks
                residuals = self.inverse_transform(transformed_coefficients)

                # we have to get the prediciton related data (mv and mode)
                predictions = self.differential_decoding(frame_marker, differential)

                # Now we have two cases: I-frame and P-frame, configurable settings
                # Inter prediction for P-frame and intra prediction for I-frame
                # For interprediciton we have motion vectors and find the prediciton blocks from last frame
                # For intra prediction, we have mode and prediciton blocks is the same block
                block_index = 0
                reference_frame = decoded_frame.copy()
                # decoded_frame = np.full((self.frame_height, self.frame_width), 128, dtype=np.uint8)

                # TODO: determine I-frame or P-frame from the first value in ...
                if frame_marker == 1:
                    for y in range(0, frame_height, block_size):
                        for x in range(0, frame_width, block_size):
                            residual = residuals[block_index]
                            mode = predictions[block_index]
                            if mode == 0:
                                # horizontal mode intra prediciton
                                if x == 0:
                                    vertical_arr = np.full((block_size, 1), 128, dtype=np.uint8)
                                else:
                                    vertical_arr = reference_frame[y:y + block_size, x - 1:x]

                                prediction_block = np.repeat(vertical_arr, block_size, axis=1)

                            elif mode == 1:
                                # vertical mode intra prediction
                                if y == 0:
                                    horizontal_arr = np.full((1, block_size), 128, dtype=np.uint8)
                                else:
                                    horizontal_arr = reference_frame[y - 1:y, x:x + block_size]
                                prediction_block = np.repeat(horizontal_arr, block_size, axis=0)

                            decoded_block = prediction_block + residual

                            # reconstruct Y-only-reconstructed file
                            decoded_frame[y:y + block_size,
                            x:x + block_size] = decoded_block.clip(0, 255).astype(np.uint8)
                            block_index += 1
                elif frame_marker == 0:
                    for y in range(0, frame_height, block_size):
                        for x in range(0, frame_width, block_size):
                            residual = residuals[block_index]

                            mv_y, mv_x, ref_index = predictions[block_index]
                            ref_index = min(max(ref_index, 0), len(self.reference_frames) - 1)

                            reference_height = y + mv_y
                            reference_width = x + mv_x

                            # add every approximated residual values to predictor block
                            prediction_block = reference_frame[reference_height: reference_height + block_size,
                                               reference_width:reference_width + block_size]
                            decoded_block = prediction_block + residual

                            # reconstruct Y-only-reconstructed file
                            decoded_frame[y:y + block_size, x:x + block_size] = decoded_block.clip(0, 255).astype(
                                np.uint8)
                        block_index += 1

                y_data = decoded_frame.tobytes()
                f_output.write(y_data)

                # edit part 1 add line 1
                self.update_reference_frames(decoded_frame)
                frame_index += 1
            except pickle.UnpicklingError as e:
                print('decode completed')
                break
            except Exception as e:
                break
                print(f"An unexpected error occurred: {e}")

        f_mdiff.close()
        f_qtc.close()
        f_output.close()