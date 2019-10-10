function MovingObjectDetection()
%create system objects used for reading video,
%detecting moving objects and displaying results
obj = setupSystemObjects();
tracks = initializeTracks(); %create and empty array of tracks
nextId = 1; %ID of the next track

%detect moving objects, and track them across video frames
while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask]=detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections]=detectionToTrackAssignment();
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    displayTrackingResults();  
end
    function obj = setupSystemObjects()
        %initialize video I/O
        %create objects for reading a video from a file, drawing tracked
        %objects in each frame, and playing the video
        
        %create a video file reader
        obj.reader = vision.VideoFileReader('Fish Swim.mp4');
        %video player to display the foreground mask
        obj.maskPlayer = vision.VideoPlayer('Position', [740,400,700,400]);
        %video player to display the video
        obj.videoPlayer = vision.VideoPlayer('Position', [20,400,700,400]);
        
        %create System objects for foreground detection and blob analysis
        %the foreground detector is used to segment moving objects from the
        %background. it outputs a binary mask, where the pixel value of 1
        %corresponds to the foreground and value of 0 corresponds to the
        %background
        obj.detector = vision.ForegroundDetector('NumGaussians',3,...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
        
        %connected groups of foreground pixels are likely to correspond to
        %moving objects. the blob analysis System object is used to find
        %such groups (called 'blobs' or 'connected components'), and
        %compute their characteristics, such as area, centroid, and the
        %bounding box
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true,...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
    end

    function tracks = initializeTracks()
        %create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end

    function frame = readFrame()
        frame = obj.reader.step();
    end

    function [centroids, bboxes, mask] = detectObjects(frame)
        %detect foreground
        mask = obj.detector.step(frame);
        %apply morphological operations to remove noise and fill in holes
        mask = imopen(mask, strel('rectangle',[3,3]));
        mask = imclose(mask, strel('rectangle', [15,15]));
        mask = imfill(mask, 'holes');
        
        %perform blob analysis to find connected components
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
        
    end

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            predictedCentroid = predict(tracks(i).kalmanFilter);
            %shift the bounding box so that its center is at the predicted
            %location
            predictedCentroid = int32(predictedCentroid)-bbox(3:4)/2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

    function [assignments, unassignedTracks, unassignedDetections]=...
            detectionToTrackAssignment()
       
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        %compute the cost of assigning each detection to each track
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i,:) = distance(tracks(i).kalmanFilter, centroids);
        end
        %solve the assignment problem
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment); 
    end
    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
           trackIdx = assignments(i,1);
           detectionIdx = assignments(i,2);
           centroid = centroids(detectionIdx, :);
           bbox = bboxes(detectionIdx, :);
           
           %correct the estimate of the object's location using the new
           %detection
           correct(tracks(trackIdx).kalmanFilter, centroid);
           
           %replace predicted bounding box with detected bounding box
           tracks(trackIdx).bbox = bbox;
           
           %update track's age
           tracks(trackIdx).age = tracks(trackIdx).age+1;
           
           %update visibility
           tracks(trackIdx).totalVisibleCount = ...
               tracks(trackIdx).totalVisibleCount + 1;
        
           tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...,
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

    function deleteLostTracks()
       if isempty(tracks)
           return;
       end
       invisibleForTooLong = 20;
       ageThreshold = 8;
       
       %compute the fraction of the track's age for which it was visible
       ages = [tracks(:).age];
       totalVisibleCounts = [tracks(:).totalVisibleCount];
       visibility = totalVisibleCounts./ ages;
       
       %find the indices of 'lost' tracks
       lostInds = (ages < ageThreshold & visibility < 0.6) | ...
           [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
       
       %delete lost tracks 
       tracks = tracks(~lostInds);
    end

    function createNewTracks()
        centroids = centroids(unassignedDetections,:);
        bboxes = bboxes(unassignedDetections, :);
        
        for i=1:size(centroids,1)
            centroid = centroids(i,:);
            bbox = bboxes(i,:);
            
            %create a Kalman filter object
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            
            %create a new track
            newTrack = struct (...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            %add it to the array of tracks
            tracks(end+1) = newTrack;
            
            %increment the next id 
            nextId = nextId + 1;
        end
    end

    function displayTrackingResults()
       %convert the frame and the mask to uint8 RGB
       frame = im2uint8(frame);
       mask = uint8(repmat(mask, [1,1,3])) .* 255;
       minVisibleCount = 8;
       
       if ~isempty(tracks)
          %noisy detections tend to result in short-lived tracks
          %only display tracks that have been visible for more than a
          %minimum number of frames
          reliableTrackInds = [tracks(:).totalVisibleCount] > minVisibleCount;
          reliableTracks = tracks(reliableTrackInds);
          
          %display the objects. if an object has not been detected in this
          %frame, display its predicted bounding box
          if ~isempty(reliableTracks)
              %get bounding boxes
              bboxes = cat(1, reliableTracks.bbox);
              
              %get ids
              ids = int32([reliableTracks(:).id]);
              
              %create labels for objects indicating the ones for which we
              %display the predicted rather than the actual location
              labels = cellstr(int2str(ids'));
              predictedTrackInds = [reliableTracks(:).consecutiveInvisibleCount]>0;
              isPredicted = cell(size(labels));
              isPredicted(predictedTrackInds) = {'predicted'};
              labels = strcat(labels, isPredicted);
              
              %draw the objects on the frame 
              frame = insertObjectAnnotation(frame, 'rectangle', bboxes, labels);
              
              %draw the objects on the mask
              mask = insertObjectAnnotation(mask, 'rectangle', bboxes, labels);
          end
       end
    
    %display the mask and the frame
    obj.maskPlayer.step(mask);
    obj.videoPlayer.step(frame);
    end
end