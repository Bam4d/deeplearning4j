/*******************************************************************************
 * Copyright (c) 2020 Konduit K. K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.observation.transform;

import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.Random;

/**
 * A transform that is only used for debugging images.
 * <p>
 * Creates a window and displays the current state of an image during transform steps
 */
public class ImageDebugTransform implements ImageTransform {


    @Override
    public float[] query(float... floats) {
        return new float[0];
    }

    @Override
    public ImageWritable getCurrentImage() {
        return null;
    }

    final JFrame frame = new JFrame();
    final Canvas canvas = new Canvas();
    Java2DNativeImageLoader j2dNil = new Java2DNativeImageLoader();
    NativeImageLoader nativeImageLoader = new NativeImageLoader();

    public ImageDebugTransform(int frameWidth, int frameHeight) {
        frame.setSize(frameWidth+30, frameHeight+30);
        frame.setLayout(new FlowLayout());
        canvas.setVisible(true);
        canvas.setBounds(15, 15, frameWidth+30, frameHeight+30);
        frame.add(canvas);
        frame.setVisible(true);
    }

    @Override
    public ImageWritable transform(ImageWritable imageWritable, Random random) {

        int width = imageWritable.getWidth();
        int height = imageWritable.getHeight();
        int colorChannels = imageWritable.getFrame().imageChannels;

        INDArray out = null;
        try {
            out = nativeImageLoader.asMatrix(imageWritable);
        } catch (IOException e) {
            e.printStackTrace();
        }
        out = out.reshape(height, width, colorChannels).swapAxes(0,2).swapAxes(1,2).mul(255);
        INDArray compressed = out.castTo(DataType.UINT8);

        Image im = j2dNil.asBufferedImage(compressed);
        canvas.getGraphics().drawImage(im, 0, 0, width, height, null);

        return imageWritable;
    }

    @Override
    public ImageWritable transform(ImageWritable imageWritable) {
        return transform(imageWritable, new Random());
    }
}
