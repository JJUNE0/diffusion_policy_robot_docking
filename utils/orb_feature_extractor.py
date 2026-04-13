import cv2

class VisualGoalChecker:
    def __init__(self, goal_img_path, min_match=400):
        self.min_match = min_match
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.img_goal = cv2.imread(goal_img_path, cv2.IMREAD_GRAYSCALE)
        if self.img_goal is None:
            raise FileNotFoundError(f"목표 이미지를 찾을 수 없습니다.")
            
        self.kp_goal, self.des_goal = self.orb.detectAndCompute(self.img_goal, None)
        if self.des_goal is None:
            raise ValueError("목표 이미지에서 특징점을 찾을 수 없습니다.")
            
        print(f"목표 이미지 로드 완료 | 추출된 특징점 개수: {len(self.kp_goal)}")

    def check_match(self, current_img):
        if len(current_img.shape) == 3:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            
        kp_curr, des_curr = self.orb.detectAndCompute(current_img, None)
        
        inlier_count = 0
        good_matches = []
        if des_curr is not None:
            matches = self.bf.match(des_curr, self.des_goal)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            inlier_count = len(good_matches)
            
        is_reached = inlier_count >= self.min_match

        return is_reached, inlier_count
