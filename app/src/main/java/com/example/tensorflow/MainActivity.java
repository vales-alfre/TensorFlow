package com.example.tensorflow;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.example.tensorflow.ml.ModelUnquant;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST = 100;
    private static final int STORAGE_PERMISSION_REQUEST = 101;
    private static final int CAMERA_CAPTURE_REQUEST = 1;
    private static final int GALLERY_CAPTURE_REQUEST = 2;

    private TextView resultTextView, confidenceTextView;
    private ImageView previewImageView;
    private Button captureButton;
    private static final int IMAGE_DIMENSION = 224;
    private static final String[] LABELS = {"Shakira", "Ganador del mundial Lionel Messi","Mister Champions Cr7","El Padre de Valverde Moises Caicedo"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initializeUIComponents();
        configureCaptureButton();
    }

    private void initializeUIComponents() {
        resultTextView = findViewById(R.id.result);
        confidenceTextView = findViewById(R.id.confidence);
        previewImageView = findViewById(R.id.imageView);
        captureButton = findViewById(R.id.button);
    }

    private void configureCaptureButton() {
        captureButton.setOnClickListener(view -> launchCamera(null));
    }

    private void launchCamera(View view)  {
        if (isPermissionGranted(Manifest.permission.CAMERA, CAMERA_PERMISSION_REQUEST)) {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (cameraIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(cameraIntent, CAMERA_CAPTURE_REQUEST);
            } else {
                Toast.makeText(this, "No camera app available", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private boolean isPermissionGranted(String permission, int requestCode) {
        if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{permission}, requestCode);
            return false;
        }
        return true;
    }

    private void processImage(Bitmap image) {
        try {
            ByteBuffer imageByteBuffer = convertImageToByteBuffer(image);
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());
            TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, IMAGE_DIMENSION, IMAGE_DIMENSION, 3}, DataType.FLOAT32);
            inputBuffer.loadBuffer(imageByteBuffer);
            ModelUnquant.Outputs outputs = model.process(inputBuffer);
            TensorBuffer outputBuffer = outputs.getOutputFeature0AsTensorBuffer();
            displayRecognitionResults(outputBuffer.getFloatArray());
            model.close();
        } catch (IOException e) {
            throw new RuntimeException("Error initializing the model", e);
        }
    }

    private ByteBuffer convertImageToByteBuffer(Bitmap image) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_DIMENSION * IMAGE_DIMENSION * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] pixelValues = new int[IMAGE_DIMENSION * IMAGE_DIMENSION];
        image.getPixels(pixelValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
        int pixelIndex = 0;
        for (int i = 0; i < IMAGE_DIMENSION; ++i) {
            for (int j = 0; j < IMAGE_DIMENSION; ++j) {
                int pixelValue = pixelValues[pixelIndex++];
                byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((pixelValue & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }

    private void displayRecognitionResults(float[] confidences) {
        int maxIndex = 0;
        float maxConfidence = 0;
        for (int i = 0; i < confidences.length; i++) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i];
                maxIndex = i;
            }
        }
        resultTextView.setText(LABELS[maxIndex]);

        StringBuilder confidenceReport = new StringBuilder();
        for (int i = 0; i < LABELS.length; i++) {
            confidenceReport.append(String.format("%s: %.1f%%\n", LABELS[i], confidences[i] * 100));
        }
        confidenceTextView.setText(confidenceReport.toString());
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST || requestCode == STORAGE_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (requestCode == CAMERA_PERMISSION_REQUEST) {
                    launchCamera(null);
                }
            }
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            Bitmap image = null;
            if (requestCode == CAMERA_CAPTURE_REQUEST) {
                image = (Bitmap) data.getExtras().get("data");
            } else if (requestCode == GALLERY_CAPTURE_REQUEST) {
                image = retrieveImageFromData(data);
            }
            if (image != null) {
                showImageAndProcess(image);
            }
        }
    }

    private Bitmap retrieveImageFromData(Intent data) {
        try {
            return MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
        } catch (IOException e) {
            throw new RuntimeException("Error retrieving image from gallery", e);
        }
    }

    private void showImageAndProcess(Bitmap image) {
        int smallerDimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, smallerDimension, smallerDimension);
        previewImageView.setImageBitmap(image);
        image = Bitmap.createScaledBitmap(image, IMAGE_DIMENSION, IMAGE_DIMENSION, false);
        processImage(image);
    }
}
