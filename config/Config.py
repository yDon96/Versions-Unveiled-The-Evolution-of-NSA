import yaml


class Config:
    __DEFAULT_CONFIG_PATH = 'config.yml'

    def __init__(self):
        """ Initialize the Config object to default values. """
        #
        self.config_file_path = self.__DEFAULT_CONFIG_PATH

        # Env settings
        self.algorithm = 'NSA'

        # Dataset settings
        self.dataset_filename = 'input.csv'
        self.normalize_dataset = True
        self.shuffle_dataset = True
        self.dataset_shuffle_seeds = [0]
        self.train_split_percentage = 0.2

        # NSA settings
        self.nsa_seeds = [0]
        self.nsa_detectors_nr = 1000
        self.nsa_radius = [1]
        self.allowed_intersection = 1
        self.allowed_intersection_increment = 0.1
        self.patience = 100

        # VNSA settings
        self.self_radius = [0.3]
        self.alpha = 0.05
        self.non_self_area_percentage = 0.99

        # Others settings
        self.date_format = None
        self.input_folder = ''
        self.output_folder = ''
        self.show_plot = False
        self.shutdown_on_end = False
        self.verbose = False

    def load(self, path=__DEFAULT_CONFIG_PATH):
        self.config_file_path = path
        with open(path, 'r') as file:
            loaded = yaml.safe_load(file)

        # Env settings
        self.algorithm = loaded['env']['algorithm']
        self.nsa_seeds = loaded['env']['nsa_seeds']
        self.nsa_detectors_nr = loaded['env']['nsa_detectors_nr']

        # Dataset settings
        self.dataset_filename = loaded['dataset']['filename']
        self.normalize_dataset = loaded['dataset']['normalize']
        self.shuffle_dataset = loaded['dataset']['shuffle']
        self.dataset_shuffle_seeds = loaded['dataset']['shuffle_seeds']
        self.train_split_percentage = loaded['dataset']['train_split_percentage']

        # NSA settings
        self.nsa_radius = loaded['NSA']['radius']
        self.allowed_intersection = loaded['NSA']['allowed_intersection']
        self.allowed_intersection_increment = loaded['NSA']['allowed_intersection_increment']
        self.patience = loaded['NSA']['patience']

        # VNSA settings
        self.self_radius =loaded['VNSA']['self_radius']
        self.alpha = loaded['VNSA']['alpha']
        self.non_self_area_percentage = loaded['VNSA']['non_self_area_percentage']

        # Others settings
        self.date_format = loaded['others']['date_format']
        self.input_folder = loaded['others']['input_folder']
        self.output_folder = loaded['others']['output_folder']
        self.show_plot = loaded['others']['show_plot']
        self.shutdown_on_end = loaded['others']['shutdown_on_end']
        self.verbose = loaded['others']['verbose']

    def reload(self):
        self.load(self.config_file_path)

    def to_string(self):
        string = f'--- Dataset Settings ---\n' \
                 f'Dataset: {self.dataset_filename} \n' \
                 f'Dataset normalized: {self.normalize_dataset} \n' \
                 f'Dataset shuffled: {self.shuffle_dataset}; with seeds: {self.dataset_shuffle_seeds} \n' \
                 f'Percentage of self sample in test set: {self.train_split_percentage} \n\n' \
                 f'--- NSA settings ---\n' \
                 f'NSA Seeds: {self.nsa_seeds} \n' \
                 f'Number of detectors wanted: {self.nsa_detectors_nr}; with radius: {self.nsa_radius} \n' \
                 f'Allowed intersection: {self.allowed_intersection}; with increment: ' \
                 f'{self.allowed_intersection_increment}; patience: {self.patience} \n\n'
        return string
