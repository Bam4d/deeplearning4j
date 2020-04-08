/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.observation.transform.legacy;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.transform.Operation;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;

public class EncodableToImageWritableTransform implements Operation<Encodable, ImageWritable> {

    private final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    private final int height;
    private final int width;
    private final int colorChannels;

    final JFrame frame = new JFrame();
    final Canvas canvas = new Canvas();
    Java2DNativeImageLoader j2dNil = new Java2DNativeImageLoader();

    public EncodableToImageWritableTransform(int height, int width, int colorChannels) {
        this.height = height;
        this.width = width;
        this.colorChannels = colorChannels;

    }

    @Override
    public ImageWritable transform(Encodable encodable) {

        // have to downcast to a float here due to this error in opencv:
        /**
         * > Unsupported depth of input image:
         * >     'VDepth::contains(depth)'
         * > where
         * >     'depth' is 6 (CV_64F)
         */

        INDArray indArray = Nd4j.create(encodable.toArray()).castTo(DataType.FLOAT).reshape(colorChannels, height, width);

        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        return new ImageWritable(nativeImageLoader.asFrame(indArray));
    }

}
