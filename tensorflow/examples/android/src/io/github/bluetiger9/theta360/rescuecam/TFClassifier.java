package io.github.bluetiger9.theta360.rescuecam;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;

import com.google.gson.Gson;

import org.webrtc.VideoFrame;
import org.webrtc.VideoTrack;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import io.github.bluetiger9.theta360.rescuecam.tensorflow.demo.Classifier;
import io.github.bluetiger9.theta360.rescuecam.tensorflow.demo.TensorFlowImageClassifier;
import io.github.bluetiger9.theta360.rescuecam.tensorflow.demo.env.ImageUtils;

public class TFClassifier {

    private static final String TAG = "TFClassifier";

    // These are the settings for the original v1 Inception model. If you want to
    // use a model that's been produced from the TensorFlow for Poets codelab,
    // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
    // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
    // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
    // the ones you produced.
    //
    // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
    // model first:
    //
    // python strip_unused.py \
    // --input_graph=<retrained-pb-file> \
    // --output_graph=<your-stripped-pb-file> \
    // --input_node_names="Mul" \
    // --output_node_names="final_result" \
    // --input_binary=true
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private static final boolean MAINTAIN_ASPECT = true;

    private final Handler handler;
    private final HandlerThread handlerThread;

    private final Bitmap croppedBitmap;

    enum State {
        IDLE, PRE_PROCESS, PROCESS
    }
    private final AtomicReference<State> state = new AtomicReference<>(State.IDLE);

    private final Classifier classifier;

    private final Map<String, BySize> bySize = new HashMap<>();

    private final Consumer<String> detectionCallback;

    private static class BySize {
        final int[] rgbBytes;
        final int width;
        final int height;
        final int rotation;

        final Matrix frameToCropTransform;

        final Bitmap rgbFrameBitmap;

        BySize(int width, int height, int rotation) {
            this.width = width;
            this.height = height;
            this.rotation = rotation;
            rgbBytes = new int[width * height];

            rgbFrameBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

            frameToCropTransform = ImageUtils.getTransformationMatrix(
                    width, height,
                    INPUT_SIZE, INPUT_SIZE,
                    rotation, MAINTAIN_ASPECT);
        }
    }



    public TFClassifier(VideoTrack videoTrack, AssetManager assetManager, Consumer<String> detectionCallback) {
        this.detectionCallback = detectionCallback;

        videoTrack.addSink(this::onFrame);

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);

        classifier =
                TensorFlowImageClassifier.create(
                        assetManager,
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);
    }

    private void onFrame(VideoFrame videoFrame) {
        if (!state.compareAndSet(State.IDLE, State.PRE_PROCESS)) {
            Log.d(TAG, "not idle");
            return;
        }

        Log.d(TAG, "Got Video Frame. rot=" + videoFrame.getRotation());

        VideoFrame.I420Buffer i420Buffer = videoFrame.getBuffer().toI420();

        BySize b = bySize(i420Buffer.getWidth(), i420Buffer.getHeight(), videoFrame.getRotation());

        ImageUtils.convertYUV420ToARGB8888(
                bufferToArray(null, i420Buffer.getDataY()),
                bufferToArray(null, i420Buffer.getDataU()),
                bufferToArray(null, i420Buffer.getDataV()),
                i420Buffer.getWidth(),
                i420Buffer.getHeight(),
                i420Buffer.getStrideY(),
                i420Buffer.getStrideU(),
                1 /*i420Buffer.getStrideV()*/,
                b.rgbBytes);

        b.rgbFrameBitmap.setPixels(b.rgbBytes, 0, b.width, 0, 0, b.width, b.height);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(b.rgbFrameBitmap, b.frameToCropTransform, null);

        handler.post(() -> {
            if (!state.compareAndSet( State.PRE_PROCESS,  State.PROCESS)) {
                Log.d(TAG, "not pre-process");
                return;
            }

            recognize();

            if (!state.compareAndSet(State.PROCESS, State.IDLE)) {
                Log.d(TAG, "not process");
                return;
            }
        });
    }

    private void recognize() {
        final long startTime = SystemClock.uptimeMillis();
        final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
        long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        Log.i(TAG, String.format("Detect: %s (time=%dms)", results, lastProcessingTimeMs));

        final String json = new Gson().toJson(results);
        detectionCallback.accept(json);
    }

    BySize bySize(int width, int height, int rotation) {
        return bySize.computeIfAbsent(
                width + "x" + height + "@" + rotation,
                _any -> new BySize(width, height, rotation));
    }

    private static byte[] bufferToArray ( byte[] b, ByteBuffer bb){
        int remaining = bb.remaining();
        if (b == null || remaining > b.length) {
            b = new byte[remaining];
        }
        bb.get(b, 0, b.length);
        return b;
    }


}
