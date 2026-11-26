import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

class OnnxProbe {
    public static void main(String[] args) throws Exception {
        var environment = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
        try (var options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(2);
            try (var session = environment.createSession(args[0], options)) {
                System.err.println("Input: " + session.getInputInfo());
                System.err.println("Output: " + session.getOutputInfo());
                System.err.println("Metadata: " + session.getMetadata());
                if (args.length == 1) return;

                var node = session.getInputInfo().values().iterator().next();
                var shape = Arrays.stream(args[2].split(",")).mapToLong(Long::parseLong).toArray();
                var modelShape = ((TensorInfo) node.getInfo()).getShape();
                if (session.getNumInputs() != 1 || modelShape.length != shape.length) {
                    throw new IllegalArgumentException("Unexpected input rank or count");
                }
                for (int axis = 0; axis < shape.length; axis++) {
                    if (modelShape[axis] >= 0 && modelShape[axis] != shape[axis]) {
                        throw new IllegalArgumentException("Input dimensions do not match");
                    }
                }
                if (args.length == 4 && args[3].equals("--batch")) {
                    for (String line : Files.readAllLines(Path.of(args[1]))) {
                        var fields = line.split("\t", -1);
                        if (fields.length != 5) {
                            throw new IllegalArgumentException("Expected label, clip ID, tensor path, split, signer ID");
                        }
                        var scores = infer(environment, session, node.getName(), shape, Path.of(fields[2]));
                        var row = new StringBuilder(String.join("\t", fields[0], fields[1], fields[3], fields[4]));
                        for (float score : scores) row.append('\t').append(score);
                        System.out.println(row);
                    }
                } else {
                    System.out.println(Arrays.toString(infer(environment, session, node.getName(), shape, Path.of(args[1]))));
                }
            }
        }
    }

    private static float[] infer(
        OrtEnvironment environment, OrtSession session, String input, long[] shape, Path path
    ) throws Exception {
        var bytes = Files.readAllBytes(path);
        var values = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        try (var tensor = OnnxTensor.createTensor(environment, values, shape);
             var results = session.run(Map.of(input, tensor))) {
            var output = ((OnnxTensor) results.get(0)).getFloatBuffer();
            var scores = new float[output.remaining()];
            output.get(scores);
            return scores;
        }
    }
}
