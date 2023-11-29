class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        # TODO - TP 1
        
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO - TP 1

        if s not in self.str_to_id:
            new_id = len(self.str_to_id)
            self.str_to_id[s] = new_id
            self.id_to_str.append(s)

        return self.str_to_id[s]

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO - TP 1

        return self.id_to_str[i]
    
    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        # TODO - TP 1
        
        if isinstance(key, int):
            return self.__get_str(key)
        elif isinstance(key, str):
            return self.__get_id(key)
        else:
            raise KeyError


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    # TODO - TP 2

    # Sumber = https://stackoverflow.com/questions/40527584/what-is-the-fastest-algorithm-for-intersection-of-two-sorted-lists
    #          Modifikasi dari TP 1

    merged_sorted_list = []
    index_1 = 0
    index_2 = 0

    while index_1 < len(posts_tfs1) and index_2 < len(posts_tfs2):
        doc_id1, tf1 = posts_tfs1[index_1]
        doc_id2, tf2 = posts_tfs2[index_2]
        
        if doc_id1 == doc_id2:
            merged_sorted_list.append((doc_id1, tf1 + tf2))
            index_1 += 1
            index_2 += 1
        elif doc_id1 < doc_id2:
            merged_sorted_list.append((doc_id1, tf1))
            index_1 += 1
        else:
            merged_sorted_list.append((doc_id2, tf2))
            index_2 += 1

    # Menambahkan sisa elemen dari posts_tfs1
    while index_1 < len(posts_tfs1):
        doc_id1, tf1 = posts_tfs1[index_1]
        merged_sorted_list.append((doc_id1, tf1))
        index_1 += 1

    # Menambahkan sisa elemen dari posts_tfs2
    while index_2 < len(posts_tfs2):
        doc_id2, tf2 = posts_tfs2[index_2]
        merged_sorted_list.append((doc_id2, tf2))
        index_2 += 1

    return merged_sorted_list