import pandas as pd

from sklearn.model_selection import train_test_split

from openfl.federated import KerasDataLoader


class MLPDataLoader(KerasDataLoader):
    def __init__(
        self,
        collaborator_count: int,
        data_path: str,
        batch_size: int,
        **kwargs
    ) -> None:
        """Instantiate the data object.

        Args:
            data_path: The file path to the data Returns:
            batch_size: The batch size of the data loader tuple: shape of an example feature array
            **kwargs: Additional arguments, passed to super init and load_mnist_shard

        Returns:
           none
        """

        change_df = pd.read_csv('./datasets/group_people.csv')
        bots_df = pd.read_csv('./datasets/bots.csv')
        users_df = pd.read_csv('./datasets/users.csv')
        data = pd.concat([bots_df, users_df, change_df])

        source_features = ['has_photo', 'has_mobile', 'is_friend', 'can_post', 'can_see_all_posts',
                           'can_see_audio', 'can_write_private_message', 'can_send_friend_request',
                           'can_be_invited_group', 'followers_count', 'blacklisted', 'blacklisted_by_me',
                           'is_favorite', 'is_hidden_from_feed', 'common_count', 'university', 'faculty',
                           'graduation', 'relation', 'verified', 'deactivated', 'friend_status', 'can_access_closed',
                           'is_closed', 'city_id', 'country_id', 'last_seen_platform', 'last_seen_time',
                           'interests_bool', 'books_bool', 'tv_bool', 'quotes_bool', 'about_bool',
                           'games_bool', 'movies_bool', 'activities_bool', 'music_bool', 'mobile_phone_bool',
                           'home_phone_bool', 'site_bool', 'status_bool', 'university_bool',
                           'university_name_bool', 'faculty_bool', 'faculty_name_bool', 'graduation_bool',
                           'home_town_bool', 'relation_bool', 'personal_bool', 'universities_bool',
                           'schools_bool', 'occupation_bool', 'education_form_bool', 'education_status_bool',
                           'relation_partner_bool', 'skype_bool', 'twitter_bool', 'livejournal_bool',
                           'instagram_bool', 'facebook_bool', 'facebook_name_bool', 'relatives_in_friends_bool',
                           'change_nickname_bool', 'partner_in_friends_bool', 'partner_in_friends_bool',
                           'posts_count', 'users_subscriptions_count', 'groups_subscriptions_count',
                           'albums_count', 'audios_count', 'gifts_count', 'pages_count', 'photos_count',
                           'subscriptions_count', 'videos_count', 'video_playlists_count',
                           'subscriptions_followers_coef',
                           'subscriptions_followers_coef_norm', 'friends_count']

        self.batch_size = batch_size
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            data[source_features],
            data['bots'],
            test_size=0.2
        )
        self.X_train.reset_index(drop=True)
        self.X_valid.reset_index(drop=True)
        self.y_train.reset_index(drop=True)
        self.y_valid.reset_index(drop=True)
