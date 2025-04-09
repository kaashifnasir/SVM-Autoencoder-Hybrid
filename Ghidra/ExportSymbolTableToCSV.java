import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Program;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.symbol.ReferenceManager;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSpace;
import ghidra.util.task.TaskMonitor;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;

public class ExportSymbolTableToCSV extends GhidraScript {
    @Override
    public void run() throws Exception {
        Program program = getCurrentProgram();
        SymbolTable symbolTable = program.getSymbolTable();
        ReferenceManager referenceManager = program.getReferenceManager();
        File programFile = new File(program.getExecutablePath());
        String programHash = generateSHA256(Files.readAllBytes(programFile.toPath()));
        String fileName = programHash + ".csv";
        Map<String, Integer> apiCallCounts = new HashMap<>();

        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write("sha256");
            for (Symbol symbol : symbolTable.getAllSymbols(true)) {
                Address address = symbol.getAddress();
                AddressSpace addressSpace = address.getAddressSpace();
                if (symbol.isExternal() || addressSpace.isExternalSpace()) {
                    String symbolName = symbol.getName();
                    int referenceCount = referenceManager.getReferenceCountTo(symbol.getAddress());
                    apiCallCounts.put(symbolName, apiCallCounts.getOrDefault(symbolName, 0) + referenceCount);
                }
            }
            for (String api : apiCallCounts.keySet()) {
                writer.write("," + api);
            }
            writer.write("\n");

            writer.write(programHash);
            for (String api : apiCallCounts.keySet()) {
                writer.write("," + apiCallCounts.get(api));
            }
            writer.write("\n");
            println("Symbol table successfully exported to " + fileName);
        } catch (IOException e) {
            println("Error writing to CSV: " + e.getMessage());
        }
    }

    private String generateSHA256(byte[] inputBytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = digest.digest(inputBytes);
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            println("Error generating hash: " + e.getMessage());
            return "";
        }
    }
}
