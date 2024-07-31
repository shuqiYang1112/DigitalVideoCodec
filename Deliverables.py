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

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class Encoder:
    def __init__(self, input_filepath, frame_height, frame_width, block_size, search_range,
                 I_period, QP, output_folder, nRefFrames = 1, VBSEnable=False, lambda_map=None, FMEenable=False,
                 FastME=False):
        self.input_filepath = input_filepath
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.block_size = block_size
        self.search_range = search_range
        self.I_period = I_period
        self.QP = QP
        self.output_folder = output_folder
        self.nRefFrames = nRefFrames  # Number of reference frames to keep
        self.VBSEnable = VBSEnable
        self.lambda_map = lambda_map or {22: 0.1, 27: 0.05, 32: 0.01}
        self.possible_block_sizes = [4, 8, 16] if self.VBSEnable else [self.block_size]
        self.FMEenable = FMEenable
        self.FastME = FastME
        self.latest_mv = [0, 0, 0]

        initial_reference_frame = np.full((frame_height, frame_width), 128, dtype=np.uint8)
        self.reference_frames = [initial_reference_frame]  # Initialize with the first reference frame
        self.MAE = []
        self.avg_PSNR = 0
        self.total_psnr = 0


        self.total_bitcount = 0
        self.frames_bitcount = []
        self.single_frame_bitcount = 0

        # ---------------deliverabele------------
        self.distortion = []
        self.split = 0
        self.total_block = 0
        self.splitPer = 0
        self.block_size_data = {}
        self.ref_frame_data = {}
        self.all_motion_vector = {}
        self.all_intra_data = {}
        # ---------------deliverabele------------
        max_uint16 = np.iinfo(np.uint16).max
        self.Q = np.zeros((block_size, block_size), dtype=np.uint16)
        for x in range(block_size):
            for y in range(block_size):
                val = 2 ** QP
                if x + y < block_size - 1:
                    self.Q[x][y] = min(max(val, 1), max_uint16)
                elif x + y == block_size - 1:
                    self.Q[x][y] = min(max(val * 2, 1), max_uint16)
                else:
                    self.Q[x][y] = min(max(val * 4, 1), max_uint16)

    def read_video_sequences(self, num_frames_to_process):
        frames = []
        original_y_only_filepath = self.output_folder + 'original_y_only.yuv'
        with open(self.input_filepath, 'rb') as f, open(original_y_only_filepath, 'wb') as f_original_y_only:
            for _ in range(num_frames_to_process):
                y_data = f.read(self.frame_width * self.frame_height)
                u_data = f.read(self.frame_width * self.frame_height // 4)
                v_data = f.read(self.frame_width * self.frame_height // 4)

                if not y_data or not u_data or not v_data:
                    break

                # dump the original from to Y-only files
                f_original_y_only.write(y_data)

                y = np.frombuffer(y_data, dtype=np.uint8).reshape(
                    self.frame_height, self.frame_width)
                frames.append(y)
        return frames

    def split_frame_into_blocks_with_padding(self, frame):
        # Determine the largest block size for padding
        max_block_size = max(self.possible_block_sizes)

        # Calculate the padding required to make the frame dimensions a multiple of max_block_size
        padding_height = (max_block_size - frame.shape[0] % max_block_size) % max_block_size
        padding_width = (max_block_size - frame.shape[1] % max_block_size) % max_block_size

        # Pad the frame with the constant value 128 (typical for YUV format)
        padded_frame = np.pad(frame,
                              pad_width=((0, padding_height), (0, padding_width)),
                              mode='constant',
                              constant_values=128)

        # The function now returns a single padded frame rather than a list of blocks
        return padded_frame

    # ------------------------------------------------new functions in b)-------------------------------------------------------
    def lambda_function(self, QP):
        # If the exact QP is in the map, return the corresponding lambda value directly
        if QP in self.lambda_map:
            return self.lambda_map[QP]

        # Otherwise, interpolate the lambda value based on surrounding QPs
        sorted_qps = sorted(self.lambda_map.keys())
        for i in range(len(sorted_qps) - 1):
            if sorted_qps[i] < QP < sorted_qps[i + 1]:
                # Linear interpolation
                lambda_low = self.lambda_map[sorted_qps[i]]
                lambda_high = self.lambda_map[sorted_qps[i + 1]]
                qp_low = sorted_qps[i]
                qp_high = sorted_qps[i + 1]

                # Calculate the interpolated lambda value
                lambda_value = lambda_low + (lambda_high - lambda_low) * (QP - qp_low) / (qp_high - qp_low)
                return lambda_value

        # If QP is outside the known range, use the nearest known lambda value
        if QP < min(sorted_qps):
            return self.lambda_map[min(sorted_qps)]
        if QP > max(sorted_qps):
            return self.lambda_map[max(sorted_qps)]

    def estimate_bit_cost(self, block, mv_or_mode, QP):
        # Transform the block
        transformed_block = self.transform([block])
        # Determine the size of the block
        block_size = block.shape[0]

        # Quantize the transformed block
        quantized_block = self.quantization(transformed_block)[0]
        # Flatten the quantized coefficients to a 1D array for bit cost estimation
        flat_quantized_block = quantized_block.flatten()

        # Estimate the bit cost for encoding the quantized coefficients
        bit_cost_coefficients = self.compute_bit_cost_for_coefficients(flat_quantized_block)

        # Estimate the bit cost for encoding the motion vector or mode
        bit_cost_mv_or_mode = self.compute_bit_cost_for_mv_or_mode(mv_or_mode)

        # Combine the bit costs
        total_bit_cost = bit_cost_coefficients + bit_cost_mv_or_mode

        return total_bit_cost

    def compute_bit_cost_for_coefficients(self, quantized_coefficients):
        # Perform run-length encoding on the quantized coefficients
        rle_encoded = self.RLE_sequence_encoding(quantized_coefficients)

        # Encode the RLE sequence using exponential Golomb coding
        exp_golomb_encoded = self.exponential_golomb_encoding(rle_encoded)

        # Calculate the bit cost based on the length of the encoded sequence
        bit_cost_coefficients = len(exp_golomb_encoded)

        return bit_cost_coefficients

    def compute_bit_cost_for_mv_or_mode(self, mv_or_mode):
        # Handle both individual integers and lists
        if not isinstance(mv_or_mode, list):
            mv_or_mode = [mv_or_mode]

        # Flatten mv_or_mode if it is a list of lists (for motion vectors)
        if mv_or_mode and isinstance(mv_or_mode[0], list):
            mv_or_mode = [item for sublist in mv_or_mode for item in sublist]

        # Encode the motion vector or mode using exponential Golomb coding
        exp_golomb_encoded = self.exponential_golomb_encoding(mv_or_mode)

        # Calculate the bit cost based on the length of the encoded sequence
        bit_cost_mv_or_mode = len(exp_golomb_encoded)

        return bit_cost_mv_or_mode

    def compute_rd_cost_for_block(self, block, x, y, block_size, reference_frames, mv_or_mode, QP):
        # This should incorporate both distortion (e.g., SAD) and estimated bit cost
        # For simplicity, let's use SAD for distortion and a placeholder bit cost function
        reference_block = reference_frames[-1][y:y + block_size,
                          x:x + block_size]  # Last reference frame for simplicity
        sad = np.sum(np.abs(block - reference_block))  # Sum of Absolute Differences
        bit_cost = self.estimate_bit_cost(block, mv_or_mode, QP)  # Placeholder bit cost function
        lambda_value = self.lambda_function(QP)  # Lambda value for RD optimization
        rd_cost = sad + lambda_value * bit_cost  # RD cost calculation
        return rd_cost

    def interpolate_frame(self, frame):
        height, width = frame.shape[:2]
        # For half-pixel accuracy, scale by a factor of 2
        new_size = (width * 2, height * 2)
        # Interpolating the frame using bilinear interpolation
        interpolated_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        return interpolated_frame

    def inter_predictions(self, reference_frames, block, x, y, block_size):
        best_mae = float('inf')
        best_mv = [0, 0, 0]  # Including the reference frame index

        if self.FMEenable:
            # Create interpolated frames
            interpolated_frames = [self.interpolate_frame(ref) for ref in reference_frames]
            step_size = 0.5  # For half-pixel accuracy
            scale = 2
        else:
            interpolated_frames = reference_frames
            step_size = 1  # Full-pixel accuracy
            scale = 1

        for ref_index, ref_frame in enumerate(interpolated_frames):
            if self.FastME:
                # Fast Motion Estimation - Nearest Neighbors search
                search_positions = self.get_nearest_neighbors(step_size)
            else:
                # Full search algorithm
                search_positions = [(mv_y, mv_x) for mv_y in
                                    np.arange(-self.search_range, self.search_range + step_size, step_size)
                                    for mv_x in np.arange(-self.search_range, self.search_range + step_size, step_size)]

            for mv_y, mv_x in search_positions:
                ref_y = y * scale + mv_y * scale
                ref_x = x * scale + mv_x * scale
                if ref_x >= 0 and ref_x + block_size * scale <= ref_frame.shape[
                    1] and ref_y >= 0 and ref_y + block_size * scale <= ref_frame.shape[0]:
                    reference_block = ref_frame[int(ref_y):int(ref_y) + block_size * scale:int(scale),
                                      int(ref_x):int(ref_x) + block_size * scale:int(scale)]
                    mae = np.mean(np.abs(np.subtract(block, reference_block, dtype=np.int16)))
                    if mae < best_mae:
                        best_mae = mae
                        best_mv = [mv_y, mv_x, ref_index]

        return best_mv

    def get_nearest_neighbors(self, step_size):
        # Define Nearest Neighbors positions around the latest MV
        latest_mv = self.latest_mv
        neighbors = [(0, 0), (0, -step_size), (0, step_size), (-step_size, 0), (step_size, 0),
                     (-step_size, -step_size), (-step_size, step_size), (step_size, -step_size), (step_size, step_size)]
        return [(latest_mv[0] + dy, latest_mv[1] + dx) for dy, dx in neighbors]

    def intra_predictions(self, block, x, y, block_size):
        # Adjusting to handle a single block of given size

        if y == 0:
            horizontal_arr = np.full((1, block_size), 128, dtype=np.uint8)
        else:
            horizontal_arr = self.reference_frames[-1][y - 1:y, x:x + block_size]

        if x == 0:
            vertical_arr = np.full((block_size, 1), 128, dtype=np.uint8)
        else:
            vertical_arr = self.reference_frames[-1][y:y + block_size, x - 1:x]

        horizontal_mode_MAE = np.mean(np.abs(np.subtract(block, vertical_arr, dtype=np.int16)))
        vertical_mode_MAE = np.mean(np.abs(np.subtract(block, horizontal_arr, dtype=np.int16)))

        # Choose the mode that gives the lowest MAE
        mode = 0 if horizontal_mode_MAE <= vertical_mode_MAE else 1

        return mode

    def compute_residual_for_intra_block(self, block, mode, x, y, block_size):
        #         print(f"compute_residual_for_intra_block: Input block size: {block.shape}, Mode: {mode}, Block size: {block_size}")
        if mode == 0:  # Horizontal mode
            if x == 0:
                reference = np.full((block_size, 1), 128, dtype=np.uint8)
            else:
                reference = self.reference_frames[-1][y:y + block_size, x - 1:x]

            prediction_block = np.repeat(reference, block_size, axis=1)

        elif mode == 1:  # Vertical mode
            if y == 0:
                reference = np.full((1, block_size), 128, dtype=np.uint8)
            else:
                reference = self.reference_frames[-1][y - 1:y, x:x + block_size]

            prediction_block = np.repeat(reference, block_size, axis=0)

        residual_block = np.subtract(block, prediction_block, dtype=np.int16)
        if residual_block.ndim == 1:
            residual_block = np.reshape(residual_block, (block_size, block_size))
        #         print(f"compute_residual_for_intra_block: Residual block size: {residual_block.shape}")
        return residual_block

    def compute_residual_for_inter_block(self, block, mv, x, y, block_size, reference_frames):
        mv_y, mv_x, ref_index = mv
        reference_frame = reference_frames[ref_index]

        if self.FMEenable:
            # Use the interpolated frame
            interpolated_frame = self.interpolate_frame(reference_frame)
            ref_y = y * 2 + mv_y * 2  # Adjust for half-pixel
            ref_x = x * 2 + mv_x * 2  # Adjust for half-pixel
            reference_block = interpolated_frame[int(ref_y):int(ref_y) + block_size * 2:int(2),
                              int(ref_x):int(ref_x) + block_size * 2:int(2)]
        else:
            ref_y = y + mv_y
            ref_x = x + mv_x
            reference_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

        residual_block = np.subtract(block, reference_block, dtype=np.int16)
        if residual_block.ndim == 1:
            residual_block = np.reshape(residual_block, (block_size, block_size))
        return residual_block

    def compute_residual_blocks(self, frame_index, reference_frames, motion_vectors, modes, padded_frame):
        frame_height, frame_width = self.frame_height, self.frame_width
        max_block_size = max(self.possible_block_sizes)
        residuals = {}
        block_index = 0

        for y in range(0, frame_height, max_block_size):
            for x in range(0, frame_width, max_block_size):
                if frame_index % self.I_period == 0:  # I-frame
                    num_sub_blocks = len(modes[block_index])
                else:  # P-frame
                    num_sub_blocks = len(motion_vectors[block_index])

                sub_block_size = max_block_size // int(np.sqrt(num_sub_blocks))

                for sub_block_index in range(num_sub_blocks):
                    sub_y = y + (sub_block_index // (max_block_size // sub_block_size)) * sub_block_size
                    sub_x = x + (sub_block_index % (max_block_size // sub_block_size)) * sub_block_size
                    sub_block = padded_frame[sub_y:sub_y + sub_block_size, sub_x:sub_x + sub_block_size]

                    if frame_index % self.I_period == 0:  # I-frame
                        mode = modes[block_index][sub_block_index]
                        residual_block = self.compute_residual_for_intra_block(sub_block, mode, sub_x, sub_y,
                                                                               sub_block_size)
                    else:  # P-frame
                        mv = motion_vectors[block_index][sub_block_index]
                        residual_block = self.compute_residual_for_inter_block(sub_block, mv, sub_x, sub_y,
                                                                               sub_block_size, reference_frames)

                    residuals[(block_index, sub_block_index)] = residual_block

                block_index += 1

        return residuals

    def transform(self, residual_blocks):
        # Apply (i x i) 2D DCT transform to each (i x i) block.
        transformed_coefficients = []

        if isinstance(residual_blocks, dict):
            # Handling dictionary input
            for key in sorted(residual_blocks.keys()):
                residual = residual_blocks[key]
                coefficient = scipy.fftpack.dct(scipy.fftpack.dct(
                    residual, axis=0, norm='ortho'), axis=1, norm='ortho')
                transformed_coefficients.append(coefficient)
        elif isinstance(residual_blocks, list):
            # Handling list input
            for residual in residual_blocks:
                coefficient = scipy.fftpack.dct(scipy.fftpack.dct(
                    residual, axis=0, norm='ortho'), axis=1, norm='ortho')
                transformed_coefficients.append(coefficient)
        else:
            raise TypeError("Input to transform must be a list or a dictionary")

        return transformed_coefficients

    def quantization(self, transformed_coefficients):
        quant_trans_coefficients = []

        for TC in transformed_coefficients:
            # Get the size of the current block
            block_size = TC.shape[0]

            # Resize the quantization matrix to match the current block size
            resized_Q = self.Q[:block_size, :block_size]

            # Perform quantization
            QTC = np.round(TC / resized_Q)
            quant_trans_coefficients.append(QTC)

        return quant_trans_coefficients

    def rescaling(self, quant_trans_coefficients):
        # Inverse quantization operation
        transformed_coefficients = []
        for QTC in quant_trans_coefficients:
            # Get the size of the current block
            block_size = QTC.shape[0]

            # Resize the quantization matrix to match the current block size
            resized_Q = self.Q[:block_size, :block_size]

            TC = QTC * resized_Q
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

    def differential_encoding(self, frame_index, motion_vectors, modes):
        differential = []

        if frame_index % self.I_period == 0:  # I-frame
            prev_mode = 0
            for mode_group in modes:
                for mode in mode_group:
                    if mode == prev_mode:
                        differential.append(0)
                    else:
                        differential.append(mode - prev_mode)
                    prev_mode = mode
        else:  # P-frame
            prev_mv = [0, 0, -1]  # Initialize with -1 for the reference frame index
            for mv_group in motion_vectors:
                for mv in mv_group:
                    mv_y, mv_x, ref_index = mv
                    prev_mv_y, prev_mv_x, prev_ref_index = prev_mv
                    differential_mv = [mv_y - prev_mv_y, mv_x - prev_mv_x, ref_index - prev_ref_index]
                    differential.append(differential_mv)
                    prev_mv = [mv_y, mv_x, ref_index]

        return differential

    def reorder_coefficient_by_frequency(self, coefficients):
        reordered_coefficients = coefficients
        return reordered_coefficients

    def diagonal_scanning(self, reordered_coefficient_matrix):
        coefficient_array = []
        matrix_height, matrix_width = reordered_coefficient_matrix.shape

        for line in range(1, (matrix_height + matrix_width)):
            start_col = max(0, line - matrix_height)
            count = min(line, (matrix_width - start_col), matrix_height)
            for j in range(0, count):
                # print(start_col + j, min(matrix_height, line) - j - 1)
                coefficient_array.append(
                    reordered_coefficient_matrix[start_col + j][min(matrix_height, line) - j - 1])
        return coefficient_array

    def RLE_sequence_encoding(self, sequences):
        RLE_encoded = []

        i = 0
        while i < len(sequences):
            if sequences[i] != 0:
                temp = []
                non_zero_length = 0
                while i < len(sequences) and sequences[i] != 0:
                    non_zero_length += 1
                    temp.append(sequences[i])
                    i += 1

                RLE_encoded.append(non_zero_length * -1)
                RLE_encoded.extend(temp)
            else:
                zero_length = 0
                while i < len(sequences) and sequences[i] == 0:
                    zero_length += 1
                    i += 1
                if i < len(sequences):
                    RLE_encoded.append(zero_length)
                else:
                    RLE_encoded.append(0)

        return RLE_encoded

    def exponential_golomb_encoding(self, values):
        sequences = ""
        for value in values:
            if value <= 0:
                mapped_value = -2 * value
            else:
                mapped_value = 2 * value - 1
            binary_plus_one = bin(int(mapped_value) + 1)[2:]
            padding_zero = len(binary_plus_one) - 1
            sequences += '0' * padding_zero + binary_plus_one
        return sequences

    def compute_bitcount(self, value):
        if value == 0:
            return 1
        else:
            return 3 + 2 * int(np.floor(np.log2(np.abs(value))))

    def entropy_encoding_predictions(self, frame_index, predictions, f):
        if frame_index % self.I_period == 0:
            prediction_info_arr = [1] + predictions
        else:
            prediction_info_arr = [0]
            for motion_vector in predictions:
                prediction_info_arr += motion_vector

        self.total_bitcount += sum([self.compute_bitcount(value) for value in prediction_info_arr])
        self.single_frame_bitcount += sum([self.compute_bitcount(value) for value in prediction_info_arr])

        entropy_encoded_sequences = self.exponential_golomb_encoding(
            prediction_info_arr)
        # sequences_bytes = bytes(
        #     int(sequences[i:i+8], 2) for i in range(0, len(sequences), 8))
        # f.write(sequences_bytes)
        np.save(f, entropy_encoded_sequences)

    def entropy_encoding_coefficients(self, coefficients, f):
        entropy_encoded_sequences = ""
        for coefficient in coefficients:
            reordered_coefficients = self.reorder_coefficient_by_frequency(
                coefficient)
            coefficient_array = self.diagonal_scanning(reordered_coefficients)
            RLE_encoded_array = self.RLE_sequence_encoding(coefficient_array)

            self.total_bitcount += sum([self.compute_bitcount(value) for value in RLE_encoded_array])
            self.single_frame_bitcount += sum([self.compute_bitcount(value) for value in RLE_encoded_array])

            sequences = self.exponential_golomb_encoding(RLE_encoded_array)
            entropy_encoded_sequences += sequences
            # entropy_encoded_sequences.append(sequences)
        # sequences_bytes = bytes(
        #     entropy_encoded_sequences + '\n', encoding='utf8')
        # f.write(sequences_bytes)
        np.save(f, entropy_encoded_sequences)

    def reconstruct_inter_block(self, residual_block, mv, x, y, block_size, reference_frames):
        mv_y, mv_x, ref_index = mv
        reference_frame = reference_frames[ref_index]

        if self.FMEenable:
            # Use the interpolated frame
            interpolated_frame = self.interpolate_frame(reference_frame)
            ref_y = y * 2 + mv_y * 2  # Adjust for half-pixel
            ref_x = x * 2 + mv_x * 2  # Adjust for half-pixel
            reference_block = interpolated_frame[int(ref_y):int(ref_y) + block_size * 2:int(2),
                              int(ref_x):int(ref_x) + block_size * 2:int(2)]
        else:
            ref_y = y + mv_y
            ref_x = x + mv_x
            reference_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

        if residual_block.ndim == 1:
            residual_block = np.reshape(residual_block, (block_size, block_size))
        reconstructed_block = reference_block + residual_block
        return reconstructed_block.clip(0, 255).astype(np.uint8)

    def reconstruct_intra_block(self, residual_block, mode, x, y, block_size):
        #         print(f"reconstruct_intra_block: Residual block size: {residual_block.shape}, Block size: {block_size}")
        if mode == 0:  # Horizontal mode
            if x == 0:
                reference = np.full(block_size, 128, dtype=np.uint8)
            else:
                reference = self.reference_frames[-1][y:y + block_size, x - 1]

            prediction_block = np.tile(reference.reshape(-1, 1), (1, block_size))

        elif mode == 1:  # Vertical mode
            if y == 0:
                reference = np.full(block_size, 128, dtype=np.uint8)
            else:
                reference = self.reference_frames[-1][y - 1, x:x + block_size]

            prediction_block = np.tile(reference, (block_size, 1))
        if residual_block.ndim == 1 and residual_block.size == block_size * block_size:
            residual_block = np.reshape(residual_block, (block_size, block_size))
        elif residual_block.ndim == 1:
            # Handle cases where the residual block size does not match the expected size
            # This might involve additional logic to handle different sizes or raise an error
            raise ValueError(f"Unexpected residual block size: {residual_block.size}")
        reconstructed_block = prediction_block + residual_block
        return reconstructed_block.clip(0, 255).astype(np.uint8)

    def reconstruct_frame(self, reference_frames, motion_vectors, modes, quant_transformed_coefficients,
                          reconstructed_frame, index):
        frame_height, frame_width = self.frame_height, self.frame_width
        max_block_size = max(self.possible_block_sizes)
        block_index = 0
        rf = np.copy(reconstructed_frame)

        # Rescale the quantized transformed coefficients
        transformed_coefficients = self.rescaling(quant_transformed_coefficients)

        # Apply the 2D inverse DCT
        inverse_residuals = self.inverse_transform(transformed_coefficients)

        # Map transformed coefficients back to the residuals dictionary structure
        residuals = {}
        residual_index = 0
        for y in range(0, frame_height, max_block_size):
            for x in range(0, frame_width, max_block_size):
                if index % self.I_period == 0:  # I-frame
                    num_sub_blocks = len(modes[block_index])
                else:  # P-frame
                    num_sub_blocks = len(motion_vectors[block_index])

                sub_block_size = max_block_size // int(np.sqrt(num_sub_blocks))

                for sub_block_index in range(num_sub_blocks):
                    residuals[(block_index, sub_block_index)] = inverse_residuals[residual_index]
                    residual_index += 1

                block_index += 1

        # Reset block index for reconstruction
        block_index = 0
        for y in range(0, frame_height, max_block_size):
            for x in range(0, frame_width, max(self.possible_block_sizes)):
                if index % self.I_period == 0:  # I-frame
                    num_sub_blocks = len(modes[block_index])
                else:  # P-frame
                    num_sub_blocks = len(motion_vectors[block_index])

                current_block_size = max(self.possible_block_sizes) // int(np.sqrt(num_sub_blocks))
                for sub_block_index in range(num_sub_blocks):
                    sub_y = y + (sub_block_index // (
                                max(self.possible_block_sizes) // current_block_size)) * current_block_size
                    sub_x = x + (sub_block_index % (
                                max(self.possible_block_sizes) // current_block_size)) * current_block_size

                    residual_block = residuals[(block_index, sub_block_index)]

                    if index % self.I_period == 0:  # I-frame
                        mode = modes[block_index][sub_block_index]
                        reconstructed_sub_block = self.reconstruct_intra_block(residual_block, mode, sub_x, sub_y,
                                                                               current_block_size)
                    else:  # P-frame
                        mv = motion_vectors[block_index][sub_block_index]
                        reconstructed_sub_block = self.reconstruct_inter_block(residual_block, mv, sub_x, sub_y,
                                                                               current_block_size, reference_frames)

                    rf[sub_y:sub_y + current_block_size, sub_x:sub_x + current_block_size] = reconstructed_sub_block
                block_index += 1

        return rf

    def calculate_psnr(self, original_frame, reconstructed_frame):
        if original_frame.shape != reconstructed_frame.shape:
            raise ValueError("Frame shapes do not match.")

        mse = np.mean(
            (np.subtract(original_frame, reconstructed_frame, dtype=np.int16)) ** 2)
        if mse == 0:
            return float('inf')

        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    # ------------------------------------------------new method--------------------------------------------
    def update_reference_frames(self, reconstructed_frame, frame_index):
        # Ensure that reconstructed_frame is a 2D array before adding
        if len(reconstructed_frame.shape) == 2 and reconstructed_frame.shape == (self.frame_height, self.frame_width):
            if frame_index % self.I_period == 0:
                # For I-frames, reset the reference frames
                self.reference_frames = [copy.deepcopy(reconstructed_frame)]
            else:
                # For P-frames, add the new frame and maintain the size of the list
                self.reference_frames.append(copy.deepcopy(reconstructed_frame))
                if len(self.reference_frames) > self.nRefFrames:
                    self.reference_frames.pop(0)
        else:
            print("Warning: Attempted to add non-2D frame to reference_frames")


    def log_rd_optimization_details(self, x, y, best_block_size, rd_costs, best_mv_or_mode):
        print(f"Block Position: ({x}, {y})")
        print(f"Selected Block Size: {best_block_size}")
        print("RD Costs for Different Block Sizes:")
        for size, cost in rd_costs.items():
            print(f"  Block Size {size}: {cost}")
        print(f"Selected MV/Mode: {best_mv_or_mode}")
        print("--------------------------------------")

    def encode(self, num_frames_to_process):
        frame_height, frame_width = self.frame_height, self.frame_width
        reconstructed_frame = np.full((frame_height, frame_width), 128, dtype=np.uint8)

        mdiff_file = self.output_folder + 'MDiff.txt'
        qtc_file = self.output_folder + 'QTC.txt'
        output_file = self.output_folder + 'reconstructed_y_only_encoder.yuv'

        f_mdiff = open(mdiff_file, 'wb')
        f_qtc = open(qtc_file, 'wb')
        f_output = open(output_file, 'wb')
        frames = self.read_video_sequences(num_frames_to_process)
        for index, frame in enumerate(frames):
            self.single_frame_bitcount = 0
            motion_vectors = []
            modes = []
            padded_frame = self.split_frame_into_blocks_with_padding(frame)

            for y in range(0, frame_height, max(self.possible_block_sizes)):
                for x in range(0, frame_width, max(self.possible_block_sizes)):
                    rd_costs = {}  # To store RD costs for different block sizes
                    best_rd_cost = float('inf')
                    best_mv_or_mode = None
                    best_block_size = None

                    for potential_block_size in self.possible_block_sizes:
                        total_rd_cost = 0
                        sub_mvs_or_modes = []

                        sub_block_size = max(self.possible_block_sizes) // (
                                    max(self.possible_block_sizes) // potential_block_size)
                        for sub_y in range(y, y + max(self.possible_block_sizes), sub_block_size):
                            for sub_x in range(x, x + max(self.possible_block_sizes), sub_block_size):
                                sub_block = padded_frame[sub_y:sub_y + sub_block_size, sub_x:sub_x + sub_block_size]

                                if index % self.I_period == 0:  # I-frame
                                    mode = self.intra_predictions(sub_block, sub_x, sub_y, sub_block_size)
                                    sub_mvs_or_modes.append(mode)
                                    rd_cost = self.compute_rd_cost_for_block(sub_block, sub_x, sub_y, sub_block_size,
                                                                             self.reference_frames, mode, self.QP)
                                else:  # P-frame
                                    mv = self.inter_predictions(self.reference_frames, sub_block, sub_x, sub_y,
                                                                sub_block_size)
                                    self.latest_mv = mv
                                    sub_mvs_or_modes.append(mv)
                                    rd_cost = self.compute_rd_cost_for_block(sub_block, sub_x, sub_y, sub_block_size,
                                                                             self.reference_frames, mv, self.QP)

                                total_rd_cost += rd_cost

                        rd_costs[potential_block_size] = total_rd_cost

                        # Compare RD cost and select the best configuration
                        if total_rd_cost < best_rd_cost:
                            best_rd_cost = total_rd_cost
                            best_mv_or_mode = sub_mvs_or_modes
                            best_block_size = potential_block_size

                    # Store the best MVs/Modes in all_motion_vectors or all_modes
                    if index % self.I_period == 0:
                        # Store modes for I-frame
                        modes.append(best_mv_or_mode)
                        if index not in self.all_intra_data:
                            self.all_intra_data[index] = {}
                        self.all_intra_data[index][(y, x)] = best_mv_or_mode
                    else:
                        # Store motion vectors for P-frame
                        motion_vectors.append(best_mv_or_mode)
                        # ---------------deliverabele------------
                        if index not in self.ref_frame_data:
                            self.ref_frame_data[index] = {}
                        self.ref_frame_data[index][(y, x)] = self.latest_mv[2]
                        if index not in self.all_motion_vector:
                            self.all_motion_vector[index] = {}
                        self.all_motion_vector[index][(y, x)] = self.latest_mv
                    # ---------------deliverabele------------
                    if best_block_size != max(self.possible_block_sizes):
                        self.split += 1

                        if index not in self.block_size_data:
                            self.block_size_data[index] = {}
                        self.block_size_data[index][(y, x)] = best_block_size

                    self.total_block += 1
                    # ---------------deliverabele------------
            # Here you need to notice whether there needs modification to differential_encoding
            # , because the motion vectors are for blocks with different shape
            differential = self.differential_encoding(
                index, motion_vectors, modes)
            # Similarly here you need to notice whether there needs modification to entropy_encoding_predictions
            self.entropy_encoding_predictions(
                index, differential, f_mdiff)

            # **Compute residuals, perform transformation, quantization, and reconstruction(check whether they need modification)
            residuals = self.compute_residual_blocks(index, self.reference_frames, motion_vectors, modes, padded_frame)
            transformed_coefficients = self.transform(residuals)
            # **solve the problem that quantization does not pass blocksize(or maybe quantization need to be fully modified)
            quant_transformed_coefficients = self.quantization(transformed_coefficients)
            # **Perform entropy encoding for the coefficients(need modify?)
            self.entropy_encoding_coefficients(quant_transformed_coefficients, f_qtc)
            self.frames_bitcount.append(self.single_frame_bitcount)
            reconstructed_frame = self.reconstruct_frame(self.reference_frames, motion_vectors, modes,
                                                         quant_transformed_coefficients, reconstructed_frame, index)
            # Update the reference frames
            self.update_reference_frames(reconstructed_frame, index)
            # Calculate PSNR for the reconstructed frame
            psnr = self.calculate_psnr(frame, reconstructed_frame)
            self.total_psnr += psnr
            self.distortion.append(psnr)
            y_data = reconstructed_frame.tobytes()
            f_output.write(y_data)

        self.avg_PSNR = self.total_psnr / num_frames_to_process
        self.splitPer = self.split / self.total_block
        f_mdiff.close()
        f_qtc.close()
        f_output.close()