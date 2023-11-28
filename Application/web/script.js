async function openFileSelector() {
    let filePath = await eel.get_file_from_user()();  // Call Python function
    console.log('Selected file:', filePath);
    // You can now pass this file path back to Python for further processing
}