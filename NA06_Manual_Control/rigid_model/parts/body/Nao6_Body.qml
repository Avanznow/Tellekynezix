import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_001_material
        objectName: "PLASTICO_BLANCO.001"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: metalico_naranja_001_material
        objectName: "Metalico_naranja.001"
        baseColor: "#ffcc1104"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: amarillo_001_material
        objectName: "AMARILLO.001"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1
    }
    PrincipledMaterial {
        id: rojo_001_material
        objectName: "ROJO.001"
        baseColor: "#ff190405"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Body_obj
        objectName: "Nao6_Body.obj"
        Model {
            id: nao6_Body
            objectName: "Nao6_Body"
            source: "meshes/nao6_Body_mesh.mesh"
            materials: [
                plastico_BLANCO_001_material,
                metalico_naranja_001_material,
                amarillo_001_material,
                rojo_001_material
            ]
        }
    }

    // Animations:
}
