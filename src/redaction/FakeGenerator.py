import random
from faker import Faker
from typing import Tuple, List, Dict
from datetime import timedelta
from dateutil import parser


class FakeGenerator:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.fake = Faker()
        self.fake.seed_instance(random_state)

    def generate_fake_phone_mapping(self, phone_numbers: List[str]) -> Dict[str, str]:
        """
        Generates a mapping of real phone numbers to fake phone numbers.

        Args:
            phone_numbers (List[str]): List of real phone numbers.

        Returns:
            Dict[str, str]: Mapping of real phone numbers to fake phone numbers.

        This method generates fake phone numbers using the Faker library and creates a mapping
        between real phone numbers and their corresponding fake phone numbers.
        """
        fake_numbers = [self.fake.phone_number() for _ in range(len(phone_numbers))]
        mapping = {k: v for k, v in zip(phone_numbers, fake_numbers)}
        return mapping

    def generate_fake_emails_mapping(self, emails: List[str]) -> Dict[str, str]:
        """
        Generates a mapping of real email addresses to fake email addresses.

        Args:
            emails (List[str]): List of real email addresses.

        Returns:
            Dict[str, str]: Mapping of real email addresses to fake email addresses.

        This method randomly selects email generation functions from available options
        and generates fake email addresses using these functions. It then creates a mapping
        between real email addresses and their corresponding fake email addresses.
        """
        email_functions = [
            self.fake.email,
            self.fake.free_email,
            self.fake.company_email,
            self.fake.safe_email,
        ]

        sampled_functions = random.choices(email_functions, k=len(emails))
        fake_emails = [sampled_functions[i]() for i in range(len(emails))]

        mapping = {k: v for k, v in zip(emails, fake_emails)}
        return mapping

    def generate_fake_urls_mapping(self, urls: List[str]) -> Dict[str, str]:
        """
        Generates a mapping of real URLs to fake URLs.

        Args:
            urls (List[str]): List of real URLs.

        Returns:
            Dict[str, str]: Mapping of real URLs to fake URLs.

        This method generates fake URLs using the `generate_fake_urls` method (assuming it exists)
        with the number of URLs equal to the length of the input list. It then creates a mapping
        between real URLs and their corresponding fake URLs.
        """
        fake_urls = self.generate_fake_urls(num_urls=len(urls))
        mapping = {k: v for k, v in zip(urls, fake_urls)}
        return mapping

    def generate_fake_names(self, num_names: int = 100) -> List[Tuple[str, str, str]]:
        """
        Generates fake names.

        Args:
            num_names (int): Number of fake names to generate.

        Returns:
            List[Tuple[str, str, str]]: A list of tuples containing (first_name, middle_name, last_name) for each fake name.

        This method generates fake names using the Faker library for various locales.
        It shuffles the generated names and combines them into tuples of (first_name, middle_name, last_name).
        """
        locales = ["de_De", "en_GB", "en_US", "es_ES", "fr_FR"]
        first_names = []
        middle_names = []
        last_names = []

        for l in locales:
            fake = Faker(l)
            limit = (
                num_names // len(locales)
                if num_names // len(locales) != 0
                else num_names
            )
            for _ in range(limit):
                first_names.append(fake.first_name())
                middle_names.append(fake.first_name())
                last_names.append(fake.last_name())

        indices = list(range(len(first_names)))
        random.shuffle(indices)

        first_names = [first_names[i] for i in indices]
        middle_names = [middle_names[i] for i in indices]
        last_names = [last_names[i] for i in indices]
        results = [
            (first_names[i], middle_names[i], last_names[i])
            for i in range(len(first_names))
        ]
        return results

    def generate_fake_locations(self, num_locations: int = 100) -> List[str]:
        """
        Generates fake locations.

        Args:
            num_locations (int): Number of fake locations to generate.

        Returns:
            List[str]: A list of fake locations.

        This method generates fake locations using the Faker library for various locales.
        It randomly selects between city, country, and address generation functions for each locale.
        """
        random.seed(self.random_state)

        locales = ["de_De", "en_GB", "en_US", "es_ES", "fr_FR"]
        locations = []

        for l in locales:
            fake = Faker(l)
            fake.seed_instance(self.random_state)
            limit = (
                num_locations // len(locales)
                if num_locations // len(locales) != 0
                else num_locations
            )
            functions = [fake.city, fake.country, fake.address]
            choice = random.choice(functions)
            for _ in range(limit):
                locations.append(choice())

        return locations

    def generate_fake_urls(self, num_urls: int = 100) -> List[str]:
        """
        Generates fake URLs.

        Args:
            num_urls (int): Number of fake URLs to generate.

        Returns:
            List[str]: A list of fake URLs.
        """
        urls = []
        random.seed(self.random_state)
        for _ in range(num_urls):
            base_url = self.fake.url()
            path = "/".join(
                self.fake.words(nb=5)
            )  # Generate a path with 5 random words
            # Add query parameters
            query_params = "?" + "&".join(
                [f"{self.fake.word()}={self.fake.word()}" for _ in range(3)]
            )  # Generate 3 query parameters
            # Combine to make a longer URL
            long_url = f"{base_url}{path}{query_params}"
            urls.append(long_url)
        return urls


def get_mapping_for_real_name(
    real_name: str, fake_names: Tuple[str, str, str]
) -> Dict[str, str]:
    """
    Generates a mapping between real name components and their fake counterparts.

    Args:
        real_name (str): The real name to generate a mapping for.
        fake_names (Tuple[str, str, str]): A tuple containing fake first name, middle name, and last name.

    Returns:
        Dict[str, str]: A mapping between real name components and their fake counterparts.

    This function generates a mapping between real name components (first name, middle name, and last name)
    and their corresponding fake counterparts.
    """
    mapping = {}
    split_name = real_name.split()
    fake_first, fake_middle, fake_last = fake_names

    mapping[split_name[0].lower()] = fake_first
    mapping[split_name[-1].lower()] = fake_last

    if len(split_name) > 2:
        mapping[split_name[1].lower()] = fake_middle

    return mapping


def get_real_fake_name_mapping(
    real_names: List[str], fake_names: List[Tuple[str, str, str]]
) -> Dict[str, str]:
    """
    Generates a mapping between real names and their fake counterparts.

    Args:
        real_names (List[str]): List of real names.
        fake_names (List[Tuple[str, str, str]]): List of tuples containing fake first name, middle name, and last name for each real name.

    Returns:
        Dict[str, str]: A mapping between real names and their fake counterparts.

    This function generates a mapping between real names and their corresponding fake counterparts,
    by applying the `get_mapping_for_real_name` function for each real name and its corresponding fake names.
    """
    mappings = {}
    for rname, fname in zip(real_names, fake_names):
        mapping = get_mapping_for_real_name(real_name=rname, fake_names=fname)
        mappings.update(mapping)

    return mappings


def get_real_fake_entity_mapping(
    real_entity: List[str], fake_entity: List[str]
) -> Dict[str, str]:
    """
    Generates a mapping between real entities and their fake counterparts.

    Args:
        real_entity (List[str]): List of real entities.
        fake_entity (List[str]): List of fake entities.

    Returns:
        Dict[str, str]: A mapping between real entities and their fake counterparts.

    """
    mappings = {real_entity[i]: fake_entity[i] for i in range(len(real_entity))}
    return mappings


def get_real_fake_date_mapping(real_dates: List[str]) -> Dict[str, str]:
    """
    Generates a mapping between real dates and their fake counterparts.

    Args:
        real_dates (List[str]): List of real entities.

    Returns:
        Dict[str, str]: A mapping between real dates and their fake counterparts.
    """
    mappings = {}
    document_date_offset = random.randint(-365, 365)
    format_choice = random.choice(["%Y-%m-%d", "%m/%d/%Y"])
    for date_str in real_dates:
        try:
            # Attempt to parse the date using dateutil.parser
            date = parser.parse(date_str, dayfirst=False, yearfirst=False)
            event_date_offset = random.randint(-10, 10)

            # Calculate the fake date
            offset = document_date_offset + event_date_offset
            delta = timedelta(days=offset)
            fake_date = date + delta
            fake_date_str = fake_date.strftime(format_choice)

            # Add to the mappings if successfully parsed and processed
            mappings[date_str] = fake_date_str

        except ValueError:
            mappings[date_str] = "DATE REDACTED"

    sorted_mapping = sorted(
        mappings.items(), key=lambda item: len(item[0]), reverse=True
    )
    sorted_mapping = dict(sorted_mapping)
    return sorted_mapping
