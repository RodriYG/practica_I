import os
import pandas as pd
import pickle
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Establecer la clave de API de OpenAI

# Ruta para guardar la base de datos vectorial
VECTOR_DB_DIR = "vector_dbs"
METADATA_DIR = "vector_db_metadatas"

# Lista de IDs de documentos
document_ids = [
    1313
    ]

# Crear un DataFrame para guardar los resultados
results = []

# Definir el modelo y parámetros
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name='gpt-4o', temperature=0, max_tokens=250)

# Define el prompt unificado
unified_prompt = """Rol: Eres experto en análisis cualitativo.
Objetivo: Codificación deductiva de manuales de convivencia escolar de Chile en base a las categorías que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile.
Categorías:
[
    {
        "Nombre": "Prohíbe Uso Durante Horario Escolar",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles durante toda la jornada escolar. Esto incluye no permitir que los estudiantes lleven dispositivos móviles al colegio, y prohibir su uso en clases, recreos, almuerzos y cualquier otra actividad escolar.",
        "Nombre Variable": "PROHIBE_JORNADA",
        "Pregunta Guía": "¿Se prohíbe explícitamente llevar o utilizar dispositivos móviles o tecnología en el establecimiento educacional, incluyendo la prohibición de su uso en clases, recreo, almuerzo y en otras actividades escolares?",
        "Consideraciones": "Aplicable solo si la prohibición es total y no se mencionan excepciones en el documento. Esta es la restricción más estricta y abarca todas las situaciones posibles durante la jornada escolar. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'SIN_REGULACIONES'",
        "Ejemplos Positivos":
        [
            "No traer al Colegio materiales ajenos al quehacer educativo, objetos cortantes, armas blancas o de fuego, celulares, o cualquier otro sistema electrónico audible y elementos innecesarios al quehacer escolar.",
            "El estudiante no podrá portar celulares, parlantes, bazookas, mp4, notebook, netbook, cámara fotográficas o de video, tablets, juegos electrónicos, u otras herramientas tecnológicas que no sean necesarias para sus aprendizajes.",
            "El Celular será retirado a la entrada al Establecimiento y será devuelto al término de la Jornada Escolar.",
            "El uso del celular queda estrictamente prohibido al interior del establecimiento (aula, patio, comedor o cualquier dependencia del establecimiento).",
            "Está prohibido traer celular o cualquier aparato electrónico al establecimiento por parte de los estudiantes.",
            "Estará PROHIBIDO que los estudiantes asistan al colegio con celulares, tablets u otros dispositivos análogos."
        ],
        "Ejemplos Negativos":
        [
            "Uso de celulares permitido durante recreos y almuerzos, pero no en clases.",
            "Uso de celulares en actividades académicas específicas con permiso del docente.",
            "Uso de dispositivos móviles permitido si no interrumpe las clases.",
            "Queda prohibido el uso de celulares u otros aparatos electrónicos personales que interrumpan el trabajo pedagógico en el aula.",
            "El uso de celulares estará permitido en casos excepcionales y debidamente justificados, como herramienta educativa específica y bajo la supervisión del docente responsable.",
            "Uso de aparatos electrónicos durante la clase sin autorización del profesor. Ej. celular, reproductor de música, Tablet.",
            "Los apoderados no deben llamar por celular a sus estudiantes durante las horas de clase; por lo tanto, procurarán hacerlo durante los periodos de recreos."
        ]
    },
    {
        "Nombre": "Prohíbe Uso Durante Clases",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles exclusivamente durante las sesiones de clase. Los estudiantes pueden llevar los dispositivos al colegio y usarlos en otras actividades como recreos y almuerzos, pero no pueden usarlos en el aula durante el horario de clase.",
        "Nombre Variable": "PROHIBE_CLASES",
        "Pregunta Guía": "¿Se prohíbe explícitamente el uso de dispositivo móviles, celulares o tecnología solamente en clases o aula y no en el resto de la jornada, recreo o almuerzo?",
        "Consideraciones": "Aplicable solo si la prohibición es total dentro del aula y no permite excepciones durante las sesiones de clase. Esta categoría es menos estricta que 'Prohíbe Uso Durante Horario Escolar'. Es excluyente con las categorías 'PROHIBE_JORNADA', 'RESTRINGE_USO' y 'SIN_REGULACIONES'",
        "Ejemplos Positivos":
        [
            "Está prohibido el uso de objetos que no correspondan y dificulten el desarrollo de la clase, como cámaras de video o fotográficas, celulares, pendrive y otros similares.",
            "Los teléfonos celulares deben permanecer apagados y guardados durante el horario de clases. Su uso está permitido únicamente en los recreos y momentos de transición entre clases.",
            "Está prohibido mantener encendidos o usar equipos personales de audio o de telefonía móvil en clases y en toda actividad escolar.",
            "Hablar o utilizar para mensajería, navegación, juego o en general, mantener encendido un teléfono celular propio o ajeno en la sala en horario de clases."
        ],
        "Ejemplos Negativos":
        [
            "Está prohibido utilizar durante el desarrollo de la clase teléfonos celulares, mp3, mp4, máquinas fotográficas, filmadoras sin autorización del profesor que se encuentre en aula o por otro estamento de la escuela.",
            "Permiso para usar dispositivos electrónicos en actividades pedagógicas.",
            "Uso de celulares restringido durante clases, con excepciones autorizadas.",
            "Uso de celulares durante clases permitido en casos específicos autorizados por el docente."
            "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
            "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
        ]
    },
    {
        "Nombre": "Restringe Uso Durante Clases",
        "Definición": "Normativas que restringen el uso de dispositivos móviles durante las clases pero permiten ciertas excepciones. Estas normativas no prohíben el uso de dispositivos móviles de manera absoluta ni durante toda la jornada escolar ni durante las sesiones de clase.",
        "Nombre Variable": "RESTRINGE_USO",
        "Pregunta Guía": "¿Existen excepciones específicas que autorizan el uso de dispositivos móviles y tecnología en clases?",
        "Consideraciones": "Aplicable si el documento permite el uso de dispositivos móviles bajo ciertas condiciones o excepciones dentro del aula y durante las clases. Esta categoría es más flexible y permite ciertos usos autorizados. Es excluyente con las categorías 'PROHIBE_CLASES', 'PROHIBE_JORNADA' y 'SIN_REGULACIONES'",
        "Ejemplos Positivos":
        [
            "Para promover un clima escolar favorable al aprendizaje, no está permitido durante la hora de clases la utilización de aparatos electrónicos tales como; teléfonos celulares, cámaras fotográficas, aparatos reproductores de música, computadores, tablets, entre otros. El alumno podrá hacer uso de los objetos mencionados sólo con autorización del profesor.",
            "En relación al uso del celular por parte del estudiante, el establecimiento comprende que bajo ciertas circunstancias el estudiante podría hacer uso de él, siempre y cuando sea solicitado y firmado una carta de compromiso de buen uso por parte del apoderado, respetando los principios de igualdad, dignidad, inclusión y no discriminación, y que, si se ha actuado no conforme a estos derechos, quedará prohibido ingresar dicho elemento al establecimiento.",
            "Se autoriza el uso en actividades académicas y formativas de dispositivos electrónicos tales como: celulares, MP3, MP4, pendrive, notebooks y otros similares, cuando el docente a cargo lo haya autorizado para el desarrollo de alguna actividad pedagógica específica.",
            "Está prohibido utilizar celulares, equipos electrónicos personales, durante las horas de clases, sin la autorización del profesor.",
            "Se prohibe utilizar equipos tecnológicos propios y del colegio de manera responsable y solo cuando está autorizado por el docente o el educador a cargo de la actividad."
            "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
            "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
        ],
        "Ejemplos Negativos":
        [
            "Uso de celulares prohibido en todo momento sin excepciones.",
            "No se permiten dispositivos móviles dentro del establecimiento en ninguna circunstancia.",
            "Uso de tecnología móvil completamente prohibido durante la jornada escolar.",
            "Está prohibido el uso de objetos que no correspondan y dificulten el desarrollo de la clase, como cámaras de video o fotográficas, celulares, pendrive y otros similares.",
            "Hablar o utilizar para mensajería, navegación, juego o en general, mantener encendido un teléfono celular propio o ajeno en la sala en horario de clases."
        ]
    },
    {
        "Nombre": "Sin Regulación",
        "Definición": "Indica que no se encontraron regulaciones o normas específicas relacionadas con el uso de dispositivos móviles y tecnología en el documento revisado. No hay restricciones ni limitaciones sobre el uso de estos dispositivos.",
        "Nombre Variable": "SIN_REGULACION",
        "Pregunta Guía": "¿No existen secciones que regulen el uso de dispositivos móviles o tecnología?",
        "Consideraciones": "Aplicable si el documento carece completamente de normas o regulaciones sobre el uso de dispositivos móviles, indicando una ausencia total de restricciones. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'PROHIBE_CLASES'",
        "Ejemplos Positivos":
        [
            "No se encontraron regulaciones específicas sobre el uso de dispositivos móviles.",
            "El documento no menciona normas sobre el uso de teléfonos celulares.",
            "No hay regulaciones explícitas sobre dispositivos móviles en el documento.",
            "El uso de tecnología móvil no está regulado según el documento revisado.",
            "No existen secciones que regulen el uso de dispositivos móviles en el documento."
        ],
        "Ejemplos Negativos":
        [
            "Uso de dispositivos móviles estrictamente regulado durante toda la jornada escolar.",
            "Normas claras sobre el uso de dispositivos móviles en el documento.",
            "Documento incluye regulaciones explícitas sobre el uso de teléfonos celulares."
        ]
    },
    {
        "Nombre": "Establece Protocolos de Confiscación de Dispositivos",
        "Definición": "Esta categoría se refiere a los reglamentos que detallan los procedimientos a seguir cuando un dispositivo móvil es utilizado inapropiadamente o sin autorización. Incluye detalles sobre cómo y cuándo se confiscarán los dispositivos y las condiciones de su devolución.",
        "Nombre Variable": "CONFISCACION",
        "Pregunta Guía": "¿Existen normas que incluyen la confiscación de dispositivos móviles o tecnología y los procedimientos asociados?",
        "Consideraciones": "Aplicable en cualquier caso donde se mencionen reglas específicas para la confiscación de dispositivos móviles. Se incluye la entrega voluntaria o por solicitud del docente a cargo, al ingresar al colegio, al utilizarlo en clases, al interrumpir la clase",
        "Ejemplos Positivos":
        [
            "El mal uso del celular por parte del alumno(a) faculta al docente a retirarlo y posteriormente entregarlo a su apoderado personalmente. En el caso del uso pedagógico de Notebooks o Tablet dentro del establecimiento. El apoderado y el alumno(a) asumen la responsabilidad en el uso, daño o pérdida del artículo.",
            "El Celular será retirado a la entrada al Establecimiento y será devuelto al término de la Jornada Escolar.",
            "El docente solicita a alumno la entrega de artefacto tecnológico.",
            "Es importante mencionar que el alumno o alumna que porte este tipo de elementos sin autorización se procederá a su retiro y será entregado en Inspectoría General, desde allí se citará al apoderado correspondiente para que retire el artículo. En caso de reincidencia el aparato será retenido y devuelto al finalizar el año escolar.",
            "Al ser sorprendido usando estos equipos durante el desarrollo de actividades académicas, ellos serán retirados por el profesor respectivo quien consignará lo ocurrido en la hoja de vida del alumno y devolverá el equipo al estudiante al finalizar la jornada. En caso de repetirse la conducta, se informará al apoderado(a) y si esta conducta persiste se procederá a retirarlo hasta finalizar el año escolar siendo entregado al apoderado(a).",
            "La Escuela retirará estos objetos en caso de no cumplir con lo que se especifica en los puntos anteriores y a tomar las medidas formativas y disciplinarias que correspondan. Los objetos tecnológicos serán devueltos al apoderado al finalizar la jornada escolar.",
            "El establecimiento no se hará responsable de daños, pérdidas o sustracciones ocurridas fuera del horario establecido para la entrega y devolución de los dispositivos móviles."
        ],
        "Ejemplos Negativos":
        [
            "Uso de dispositivos móviles permitido siempre que no interrumpa el desarrollo de actividades escolares.",
            "No se menciona ningún procedimiento específico para la confiscación de dispositivos móviles.",
            "Se autoriza el uso en actividades académicas y formativas de dispositivos electrónicos tales como: celulares, MP3, MP4, pendrive, notebooks y otros similares, cuando el docente a cargo lo haya autorizado para el desarrollo de alguna actividad pedagógica específica.",
            "Se prohibe utilizar equipos tecnológicos propios y del colegio de manera responsable y solo cuando está autorizado por el docente o el educador a cargo de la actividad."
            "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
            "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
        ]
    },
    {
        "Nombre": "Prohíbe Grabaciones y Fotos Dentro del Establecimiento Sin Autorización",
        "Definición": "Esta categoría abarca las prohibiciones de uso de tecnología y dispositivos móviles para capturar imágenes o grabar videos dentro de las instalaciones escolares sin el permiso explícito de la administración escolar o del docente, y establece sanciones para quienes falten a esta normativa.",
        "Nombre Variable": "PROHIBE_FOTOS",
        "Pregunta Guía": "¿Se prohíbe explícitamente fotografiar y grabar sin autorización?",
        "Consideraciones": "Solo se aplica cuando en el documento se mencionan restricciones y prohibiciones de grabaciones y fotografías",
        "Ejemplos Positivos":
        [
            "No está permitido tomar fotos o videos sin autorización respectiva en ningún área de la Escuela, con el fin de respetar la integridad y privacidad de cada miembro de la comunidad educativa y de evitar publicaciones que puedan ser privadas u ofensivas.",
            "Se prohibe fotografiar pruebas, guías evaluadas, libro de clases, bases de datos del colegio o cualquier otro documento sin autorización.",
            "Está prohibido tomar fotografías a compañeros o docentes para generar burlas o memes.",
            "Está prohibido sacar fotos, videos o capturas de pantalla sin autorización, que sean o no ofensivas para sus compañeras, padres y apoderados, funcionarios y/o profesores de la Comunidad Educativa.",
            "Fotografiar, filmar en clases o grabar conversaciones con docentes, asistentes de la educación o Equipo Directivo del Colegio o estudiantes, con cualquier medio electrónico, sin autorización o contra la voluntad del tercero.",
            "Uso inadecuado del teléfono celular, como tomar fotografías o videos sin la autorización respectiva.",
            "Subir fotografías, videos, imágenes u otros a la red informática que atente contra la dignidad de las personas o que perjudique la imagen de la comunidad educativa."
        ],
        "Ejemplos Negativos":
        [
            "No hay menciones explícitas sobre la prohibición de fotos y grabaciones.",
            "El establecimiento no se responsabiliza por la pérdida o destrozo de estos objetos o cualquier otro de valor que porten los estudiantes dentro del colegio.",
            "No se prohíbe explícitamente el uso de dispositivos para grabaciones y fotos sin autorización.",
            "Se prohibe utilizar equipos tecnológicos propios y del colegio de manera responsable y solo cuando está autorizado por el docente o el educador a cargo de la actividad."
            "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
            "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
        ]
    },
    {
        "Nombre": "Regula Uso Inapropiado",
        "Definición": "Esta categoría incluye normativas que abordan el uso dispositivos móviles o tecnología de manera indebida, como el acceso a contenido inapropiado, el ciberacoso, o el uso de dispositivos para actividades disruptivas o no éticas.",
        "Nombre Variable": "USO_INAPR",
        "Pregunta Guía": "¿Se regula el tipo de uso inapropiado que los alumnos pueden darle a sus dispositivos móviles y tecnología?",
        "Consideraciones": "se aplica cuando en el documento existen reglamentosa (no solo definiciones), frente al ciberacoso, cyberbulling, distribución de material pornográfico, uso inapropiado de internet, entre otros. Se centra en el uso inapropiado y se excluye el uso apropiado",
        "Ejemplos Positivos":
        [
            "Se prohibe amenazar, atacar, injuriar o desprestigiar a un alumno o a cualquier otro integrante de la comunidad educativa a través de redes sociales, mensajes de texto, correos electrónicos, foros, servidores que almacenan videos o fotografías, sitios webs, teléfonos o cualquier otro medio tecnológico, virtual o electrónico, como también de manera verbal.",
            "Se prohibe exhibir, transmitir y/o difundir por medios cibernéticos cualquier conducta de maltrato escolar.",
            "Se prohibe hacer uso de Red Internet o de medios u objetos tecnológicos para: afectar la privacidad, la honra, ofender, amenazar, injuriar, calumniar, desprestigiar a cualquier integrante de la Comunidad Escolar, provocando daño psicológico al, o los afectados. (Ley de la Violencia Escolar 2.536)",
            "Se prohibe ejercer bullying, cyberbullying, acoso permanente a una persona.",
            "Se prohibe la utilización de medios cibernéticos o audiovisuales, para menoscabar la dignidad y honra de los/as estudiantes, funcionarios/as.",
            "Como comunidad educativa, se entenderá por maltrato escolar, cualquier acción intencional, ya sea física o psicológica, realizada en forma escrita, verbal o a través de medios tecnológicos o cibernéticos, en contra de cualquier integrante de la comunidad educativa, con independencia del lugar en que se cometa",
            "Amenazar, atacar, injuriar o desprestigiar a un alumno o a cualquier otro integrante de la comunidad educativa a través de chats, blogs, Facebook, mensajes de textos, correo electrónico, foros, servidores que almacenan videos o fotografías, sitios webs, teléfonos o cualquier otro medio tecnológico, virtual o electrónico"
        ],
        "Ejemplos Negativos":
        [
            "Uso de dispositivos móviles no está regulado en relación con actividades inapropiadas.",
            "No se mencionan restricciones específicas sobre el uso inapropiado de tecnología.",
            "Se permite el uso de celulares en casos excepcionales y debidamente justificados, como herramienta educativa específica y bajo la supervisión del docente responsable.",
            "Hablar o utilizar para mensajería, navegación, juego o en general, mantener encendido un teléfono celular propio o ajeno en la sala en horario de clases."
        ]
    },
    {
        "Nombre": "Establece Protocolos de Uso Durante Clases (Incluye Clases Online y Aula de Informática)",
        "Definición": "Esta categoría incluye normativas que definen orientaciones o protocolos para un buen uso de los dispositivos móviles y tecnología durante las clases, incluidas las clases virtuales y en aulas de informática, promoviendo un uso educativo y regulado.",
        "Nombre Variable": "PROTOCOLO_USO",
        "Pregunta Guía": "¿Existen protocolos que definen buenas prácticas u orientaciones para el buen uso de dispositivos móviles y tecnología durante clases y en otras instancias educativas?",
        "Consideraciones": "Aplicable solo si el documento incluye protocolos o reglamentos específicos sobre buenas prácticas y el uso adecuado de tecnología en entornos educativos. Se centra en el uso apropiado y se excluye el uso inapropiado. Se refiere a un conjunto de pasos a seguir para un buen uso de la tecnología en el aula o en las clases virtuales",
        "Ejemplos Positivos":
        [
            "NORMAS DE COMPORTAMIENTO DURANTE LA CLASE VIRTUAL.",
            "PROTOCOLO PROCEDIMIENTOS Y ORIENTACIONES PARA EL USO DE CELULARES Y OTROS DISPOSITIVOS MÓVILES EN EL COLEGIO.",
            "USO DE CELULAR Y APARATOS TECNOLÓGICOS.",
            "PROTOCOLO DE CONVIVENCIA DIGITAL."
        ],
        "Ejemplos Negativos":
        [
            "No se mencionan protocolos específicos para el uso de dispositivos móviles.",
            "Uso de tecnología no regulado por ningún protocolo en particular.",
            "El establecimiento no se responsabiliza por la pérdida o destrozo de estos objetos o cualquier otro de valor que porten los estudiantes dentro del colegio.",
            "Se prohibe amenazar, atacar, injuriar o desprestigiar a un alumno o a cualquier otro integrante de la comunidad educativa a través de redes sociales, mensajes de texto, correos electrónicos, foros, servidores que almacenan videos o fotografías, sitios webs, teléfonos o cualquier otro medio tecnológico, virtual o electrónico, como también de manera verbal.",
            "Se prohibe exhibir, transmitir y/o difundir por medios cibernéticos cualquier conducta de maltrato escolar.",
            "Se prohibe hacer uso de Red Internet o de medios u objetos tecnológicos para: afectar la privacidad, la honra, ofender, amenazar, injuriar, calumniar, desprestigiar a cualquier integrante de la Comunidad Escolar, provocando daño psicológico al, o los afectados. (Ley de la Violencia Escolar 2.536)",
            "Para promover un clima escolar favorable al aprendizaje, no está permitido durante la hora de clases la utilización de aparatos electrónicos tales como; teléfonos celulares, cámaras fotográficas, aparatos reproductores de música, computadores, tablets, entre otros. El alumno podrá hacer uso de los objetos mencionados sólo con autorización del profesor.",
            "En relación al uso del celular por parte del estudiante, el establecimiento comprende que bajo ciertas circunstancias el estudiante podría hacer uso de él, siempre y cuando sea solicitado y firmado una carta de compromiso de buen uso por parte del apoderado, respetando los principios de igualdad, dignidad, inclusión y no discriminación, y que, si se ha actuado no conforme a estos derechos, quedará prohibido ingresar dicho elemento al establecimiento.",
            "Se autoriza el uso en actividades académicas y formativas de dispositivos electrónicos tales como: celulares, MP3, MP4, pendrive, notebooks y otros similares, cuando el docente a cargo lo haya autorizado para el desarrollo de alguna actividad pedagógica específica."
        ]
    },
    {
        "Nombre": "Prohíbe Contacto con el Establecimiento Vía Redes Sociales",
        "Definición": "Normativas que prohíben el uso de cuentas personales o redes sociales para comunicarse con el personal docente o entidades del colegio, requiriendo el uso de medios oficiales para dicha comunicación.",
        "Nombre Variable": "PROHIBE_REDES",
        "Pregunta Guía": "¿Se prohíbe que estudiantes o apoderados usen las redes sociales u otras plataformas no institucionales para contactarse con el personal docente del establecimiento?",
        "Consideraciones": "Aplicable solo si el documento menciona restricciones específicas sobre la comunicación entre estudiantes o apoderados y el personal del colegio mediante redes sociales no institucionales.",
        "Ejemplos Positivos":
        [
            "No se permite el uso de cuentas personales en redes sociales para comunicarse con personal del colegio.",
            "Se prohíbe a Docentes, Directivos y Asistentes de la educación, la solicitud de “amistades” o “seguidores” a través de redes sociales con estudiantes o apoderados/as del Colegio. Toda vez que la relación pedagógica que prima en el establecimiento es necesariamente un vínculo formal y por tanto asimétrico, donde la figura del/la adulto(a) conlleva responsabilidades que no se extinguen por la sola existencia de un horario laboral y de forma de resguardar la privacidad de los espacios de intimidad de cada persona.",
            "En relación a las redes sociales, se prohíbe a todos los funcionarios del Colegio mantener algún tipo de conversación personal con alumnos del establecimiento por medio de estos canales virtuales (Facebook, Whatsapp, Skype, Instagram, Twitter, otros).",
            "Se debe considerar que, a modo preventivo y teniendo en cuenta el valor del espacio personal y la intimidad, los/as estudiantes no mantengan contacto a través de las redes sociales con las personas adultas que laboran en el establecimiento educativo. Sólo debería permitirse, si el profesor(a) jefe lo estima necesario, mantener un correo electrónico del curso para acoger preguntas de los alumno/as y/o Apoderados/as, así como para transmitir información importante respecto a la asignatura (MINEDUC, 27).",
            "Todo contacto por medio de redes virtuales entre los alumnos y el Colegio, incluyendo a los funcionarios, debe ser realizado a través de cuentas institucionales y no personales, por lo tanto, también queda prohibido al personal del Colegio que incluyan a los alumnos como contactos de sus redes sociales personales, salvo fines pedagógicos. El Colegio no responderá por dichos, actos, imágenes y/o situaciones relacionadas con redes personales entre sus funcionarios y familias y alumnos del Colegio que no sean a través de canales oficiales de comunicación."
        ],
        "Ejemplos Negativos":
        [
            "No hay restricciones específicas sobre el uso de redes sociales para comunicarse con personal del colegio.",
            "Uso de redes sociales permitido para comunicarse con docentes bajo ciertas condiciones.",
            "No se mencionan prohibiciones explícitas sobre el contacto en redes sociales."
        ]
    },
    {
        "Nombre": "Prohíbe Uso Horario Laboral",
        "Definición": "Normativas que regulan el uso de dispositivos móviles por parte del personal docente durante el horario laboral, enfocadas en mantener la profesionalidad y evitar distracciones.",
        "Nombre Variable": "PROHIBE_LABORAL",
        "Pregunta Guía": "¿Se prohíbe al personal docente el uso de dispositivos durante la jornada laboral (prohibición dirigida al personal)?",
        "Consideraciones": "Aplicable si el documento menciona restricciones específicas al uso de dispositivos móviles por parte del personal docente o trabajadores durante la jornada laboral.",
        "Ejemplos Positivos":
        [
            "No hablar con celular durante el desarrollo de clases u otra actividad educativa y, menos aún salir de la sala de clases para hacerlo, descuidando la protección y observación vigilante de las estudiantes, que están bajo su responsabilidad, salvo que se trate de alguna emergencia o se utilice con fines pedagógicos.",
            "El uso de instrumentos tecnológicos, tanto para estudiantes, profesores/as, directivos o asistentes de la educación está regulado.",
            "Quedan prohibidas expresamente, para cualquier funcionario del centro educativo, utilizar el teléfono celular mientras se desarrollan actividades con los estudiantes con fines pedagógicos.",
            "El (la) profesor (a) debe mantener apagado su celular mientras realiza las clases.",
            "No usar celulares durante las horas de clases frente a sus alumnos y alumnas. Así como tampoco subir a alguna red social algo que atente a la integridad de nuestra comunidad educativa."
        ],
        "Ejemplos Negativos":
        [
            "El uso de dispositivos móviles por parte del personal docente está permitido en ciertas circunstancias.",
            "No se mencionan restricciones específicas sobre el uso de celulares por parte del personal docente.",
            "Para promover prácticas pedagógicas que faciliten el aprendizaje de los estudiantes, está permitido durante la hora de clases la utilización de aparatos electrónicos, tales como: teléfonos celulares, cámaras fotográficas, aparatos reproductores de música, computadores, tablets, entre otros."
        ]
    }
]

Tareas:
1. Identificación de Acápites Relacionados al uso de tecnología y dispositivos móviles:
- Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles, tecnología y sus sanciones asociadas.
- Prioriza los siguientes términos de búsqueda: celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso, ciberbullying, objetos, devolución, entrega, retirar, confiscar, foto, videos, grabar, fotografiar, redes, sociales, facebook, whatsapp, instagram, twitter, virtuales, protocolos, autorización, llamadas, contestar, docentes, personal (considerando que a veces el prefijo ciber- es escrito cyber-).

2. Análisis y Codificación:
- Para las categorías PROHIBE_JORNADA, PROHIBE_CLASES, RESTRINGE_USO y SIN_REGULACION, son mutuamente excluyentes y es obligatorio que una y solo una de ellas se le asigne el valor 1.
- Para cada acápite identificado, determina si alguna de las siguientes categorías le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría descrita más arriba. Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.

Output:
Responde con un objeto JSON válido, no incluyas nada más en tu respuesta. 
El formato del objeto JSON debe ser el siguiente:
JSON:
{
  "PROHIBE_JORNADA": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "PROHIBE_CLASES": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "RESTRINGE_USO": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "SIN_REGULACION": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "CONFISCACION": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "PROHIBE_FOTOS": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "USO_INAPR": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "PROTOCOLO_USO": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "PROHIBE_REDES": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.,
  "PROHIBE_LABORAL": Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de cada categoría definidas más arriba. Responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.
}
"""

def create_vector_db_for_document(doc_id):
    pdf_path = f"./muestra/ReglamentoConvivencia_{doc_id}.pdf"
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
   
    if not documents:
        raise ValueError(f"No se pudieron cargar documentos del archivo {pdf_path}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    if not texts:
        raise ValueError(f"No se pudo dividir en texto el documento {pdf_path}") 

    vector_db_path = os.path.join(VECTOR_DB_DIR, f"vector_db_{doc_id}.faiss")
    metadata_path = os.path.join(METADATA_DIR, f"metadata_{doc_id}.pkl")
    
    if not os.path.exists(vector_db_path) or not os.path.exists(metadata_path):
        index = FAISS.from_documents(texts, embeddings)
        index.save_local(vector_db_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump({'document_id': doc_id}, f)
    else:
        index = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    return index, texts

# Asegurarse de que los directorios existen
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Procesar cada documento
for doc_id in document_ids:
    try:
        print(f"Procesando documento {doc_id}...")
        index, texts = create_vector_db_for_document(doc_id)
        retriever = index.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        # Realizar la consulta unificada
        print(f"Consultando documento {doc_id}...")
        llm_response = qa({"query": unified_prompt, "documents": texts})
        response = llm_response["result"]
        #print(response)

        response_json = json.loads(response)
        response_json["id"] = doc_id
        results.append(response_json)
        
    except Exception as err:
        error_response = {
            "id": doc_id,
            "PROHIBE_JORNADA": 0,
            "PROHIBE_CLASES": 0,
            "RESTRINGE_USO": 0,
            "SIN_REGULACION": 0,
            "CONFISCACION": 0,
            "PROHIBE_FOTOS": 0,
            "USO_INAPR": 0,
            "PROTOCOLO_USO": 0,
            "PROHIBE_REDES": 0,
            "PROHIBE_LABORAL": 0,
            "error": f'Exception occurred. Please try again: {str(err)}'
        }
        results.append(error_response)

# Guardar los resultados en un archivo CSV
df = pd.DataFrame(results)
df.to_csv("responses_with_categories_15.csv", index=False)

print(f"Results saved to responses_with_categories_3.csv")