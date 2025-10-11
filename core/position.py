import importlib
import os

import cv2

image_x_y_range = {

}

image_dic = {

}

initialized_image = {
    'CN': False,
    'Global_zh-tw': False,
    'Global_en-us': False,
    'Global_ko-kr': False,
    'JP': False
}


def init_image_data(self):
    global image_x_y_range
    global image_dic
    server = self.identifier
    if initialized_image[server]:
        return True
    image_dic.setdefault(server, {})
    image_x_y_range.setdefault(server, {})
    initialized_image[server] = True
    path = 'src/images/' + server + '/x_y_range'
    for file_path, child_file_name, files in os.walk(path):
        if file_path.endswith('activity'):
            continue
        for filename in files:
            if filename.endswith('py'):
                temp = file_path.replace('\\', '.')
                temp = temp.replace('/', '.')
                import_name = temp + '.' + filename.split('.')[0]
                data = importlib.import_module(import_name)
                x_y_range = getattr(data, 'x_y_range', None)
                path = getattr(data, 'path', None)
                prefix = getattr(data, 'prefix', None)
                if prefix in image_x_y_range[server]:
                    image_x_y_range[server][prefix].update(x_y_range)
                else:
                    image_x_y_range[server][prefix] = x_y_range
                for key in x_y_range:
                    img_path = 'src/images/' + server + '/' + path + '/' + key + '.png'
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        image_dic[server][prefix + '_' + key] = img

    if self.current_game_activity is not None:
        BASIC_DATA_LOCATION = f"module/activities/activity_data"
        if os.path.isfile(f"{BASIC_DATA_LOCATION}/{self.server}_event_entry_1.png") and \
            os.path.isfile(f"{BASIC_DATA_LOCATION}/{self.server}_event_entry_2.png"):
            image_dic[server]["activity_entry-1"] = cv2.imread(f"{BASIC_DATA_LOCATION}/{self.server}_event_entry_1.png")
            image_dic[server]["activity_entry-2"] = cv2.imread(f"{BASIC_DATA_LOCATION}/{self.server}_event_entry_2.png")
    self.logger.info(f"Image {server} count : {len(image_dic[server])}")
    return True


def alter_img_position(self, name, point):
    global image_x_y_range
    global image_dic
    if name in image_dic[self.identifier]:
        shape = image_dic[self.identifier][name].shape
        prefix, name = name.rsplit("_", 1)
        if image_x_y_range[self.identifier][prefix][name] is not None:
            self.logger.info("Alter position of : [ " + name + " ] --> " + str(point))
            image_x_y_range[self.identifier][prefix][name] = (point[0], point[1], point[0] + shape[1],
                                                              point[1] + shape[0])


def get_area(identifier, name):
    global image_x_y_range
    global image_dic
    prefix, name = name.rsplit("_", 1)
    if image_x_y_range[identifier][prefix][name] is None:
        return False
    return image_x_y_range[identifier][prefix][name]
