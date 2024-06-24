import os
import uuid

def write_files_to_disk(files_contents):    
    file_paths = []
    
    for content in files_contents:
        unique_filename = str(uuid.uuid4()) + ".pdf"
        path = os.path.join('app', 'temp_files', unique_filename)

        with open(path, 'wb') as file:
            file.write(content)
        
        file_paths.append(path)

    return file_paths 
